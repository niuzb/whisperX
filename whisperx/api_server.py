import base64
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from typing import Dict, Any, Optional
from dataclasses import replace

from fastapi import FastAPI, Header, BackgroundTasks, Request, Response
from pydantic import BaseModel
import torch
import uvicorn
import numpy as np

# Verify if we can import from whisperx, otherwise try relative imports
try:
    from whisperx.asr import load_model, FasterWhisperPipeline
    from whisperx.audio import load_audio
except ImportError:
    # If running from inside the package without installation
    from .asr import load_model, FasterWhisperPipeline
    from .audio import load_audio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisperx_api")

app = FastAPI()

# Global state
tasks: Dict[str, Dict[str, Any]] = {}
model_pipeline: Optional[FasterWhisperPipeline] = None
model_lock = threading.Lock()

# Configuration
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
LANGUAGE = None # Auto-detect

class AudioData(BaseModel):
    data: str

class UserData(BaseModel):
    uid: str

class RequestData(BaseModel):
    model_name: Optional[str] = None
    enable_itn: Optional[bool] = False
    enable_punc: Optional[bool] = True
    context: Optional[str] = None # JSON string with hotwords

class SubmitBody(BaseModel):
    user: UserData
    audio: AudioData
    request: RequestData

@app.on_event("startup")
def startup_event():
    global model_pipeline
    logger.info(f"Loading model {MODEL_NAME} on {DEVICE} with {COMPUTE_TYPE}...")
    try:
        model_pipeline = load_model(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            language=LANGUAGE,
            vad_method="pyannote",
            vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # We don't raise here to allow the server to start, 
        # but requests will fail. Or we should raise.
        # raise e

def process_transcription(task_id: str, audio_path: str, hotwords_str: Optional[str]):
    global model_pipeline
    try:
        if model_pipeline is None:
            raise RuntimeError("Model not loaded")

        tasks[task_id]["status"] = "processing"
        logger.info(f"Processing task {task_id}")

        # Load audio
        # load_audio returns float32 numpy array
        audio = load_audio(audio_path)

        with model_lock:
            # Handle hotwords if provided
            original_options = model_pipeline.options
            if hotwords_str:
                # Create a new options object with updated hotwords
                # We use replace from dataclasses
                model_pipeline.options = replace(original_options, hotwords=hotwords_str)
            
            try:
                # Transcribe
                # Using defaults for batch_size etc.
                result = model_pipeline.transcribe(
                    audio,
                    batch_size=16,
                    chunk_size=30,
                    print_progress=False
                )
            finally:
                if hotwords_str:
                    # Restore original options
                    model_pipeline.options = original_options
        
        # Format output to match Volcengine style
        full_text = ""
        utterances = []
        
        # result["segments"] contains the transcribed segments
        for segment in result["segments"]:
            text = segment["text"].strip()
            full_text += text + " "
            utterances.append({
                "text": text,
                "start_time": int(segment["start"] * 1000),
                "end_time": int(segment["end"] * 1000)
            })
            
        duration_ms = int(len(audio) / 16000 * 1000)
        
        tasks[task_id]["result"] = {
            "audio_info": {
                "duration": duration_ms
            },
            "result": {
                "additions": {
                    "duration": str(duration_ms)
                },
                "text": full_text.strip(),
                "utterances": utterances
            }
        }
        tasks[task_id]["status"] = "success"
        logger.info(f"Task {task_id} completed.")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
    finally:
        # Cleanup temp file
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/submit")
async def submit_task(
    body: SubmitBody, 
    background_tasks: BackgroundTasks,
    x_api_request_id: Optional[str] = Header(None, alias="X-Api-Request-Id")
):
    task_id = str(uuid.uuid4())
    
    # Decode audio
    try:
        audio_bytes = base64.b64decode(body.audio.data)
    except Exception:
        return Response(status_code=400, content="Invalid base64 audio")
    
    # Save to temp file
    # We create a temp file and pass its path to the background task
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, 'wb') as f:
        f.write(audio_bytes)
    
    # Parse hotwords from context
    hotwords_str = None
    if body.request.context:
        try:
            ctx = json.loads(body.request.context)
            # Client format: hotwords: [{word: "..."}]
            if "hotwords" in ctx and isinstance(ctx["hotwords"], list):
                words = [w["word"] for w in ctx["hotwords"] if isinstance(w, dict) and "word" in w]
                if words:
                    hotwords_str = ",".join(words)
        except Exception as e:
            logger.warning(f"Failed to parse context/hotwords: {e}")

    tasks[task_id] = {
        "status": "pending",
        "created_at": time.time()
    }
    
    background_tasks.add_task(process_transcription, task_id, temp_path, hotwords_str)
    
    # Return response compatible with Volcengine
    # Volcengine submit response usually contains the task ID in body.
    # The client looks for { xTtLogId } in response.
    return {
        "xTtLogId": task_id
    }

@app.post("/query")
async def query_task(
    x_tt_logid: Optional[str] = Header(None, alias="X-Tt-Logid")
):
    if not x_tt_logid:
         return Response(status_code=400, content="Missing X-Tt-Logid")
         
    task = tasks.get(x_tt_logid)
    if not task:
        return Response(status_code=404, content="Task not found")
        
    status = task["status"]
    if status == "success":
        return Response(
            content=json.dumps(task["result"]),
            media_type="application/json",
            headers={"X-Api-Status-Code": "20000000", "X-Api-Message": "Success"}
        )
    elif status == "failed":
        return Response(
            content=json.dumps({"error": task.get("error")}),
            media_type="application/json",
            headers={"X-Api-Status-Code": "30000000", "X-Api-Message": "Failed"}
        )
    else:
        # Pending or processing
        # Volcengine code 1000/1001? Client checks != 20000000 and not in PENDING_CODES.
        # We assume 1000 is pending.
        return Response(
            content=json.dumps({}),
            media_type="application/json",
            headers={"X-Api-Status-Code": "1000", "X-Api-Message": "Pending"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
