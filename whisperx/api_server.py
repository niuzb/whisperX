import asyncio
import base64
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import defaultdict

from fastapi import Body, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# For Clustering
import numpy as np
from sklearn.cluster import DBSCAN

app = FastAPI()

from whisperx.log_utils import get_logger
from whisperx.diarize import DiarizationPipeline, assign_word_speakers, SpeechEmbeddingPipeline
from whisperx.audio import load_audio, SAMPLE_RATE

logger = get_logger(__name__)

ASR_STATUS_SUCCESS = "20000000"
ASR_STATUS_PROCESSING = "20000001"
ASR_STATUS_IN_QUEUE = "20000002"
ASR_STATUS_NO_SPEECH = "20000003"

ASR_STATUS_INVALID_PARAMS = "45000001"
ASR_STATUS_SYSTEM_ERROR = "55000000"


class SubmitUser(BaseModel):
    uid: Optional[str] = None


class SubmitAudio(BaseModel):
    # 客户端会传 base64: requestData.audio.data
    data: Optional[str] = None
    # 兼容：如果未来改为传 URL
    url: Optional[str] = None


class SubmitRequest(BaseModel):
    # 客户端字段名是 request.model_name（来自 SpeechToTextConfig.MODEL_NAME）
    # 这里把它当作 whisper / faster-whisper 的模型名使用；不传则用默认值
    model_name: Optional[str] = None

    # 以下字段来自客户端
    enable_itn: Optional[bool] = None
    enable_punc: Optional[bool] = None
    enable_speaker_info: Optional[bool] = None  # Controls Diarization
    enable_ddc: Optional[bool] = None
    show_utterances: Optional[bool] = None
    enable_lid: Optional[bool] = None
    context: Optional[str] = None
    
    # New fields for diarization control
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


class TaskSubmission(BaseModel):
    user: Optional[SubmitUser] = None
    audio: SubmitAudio
    request: Optional[SubmitRequest] = None


class SpeakerSegment(BaseModel):
    speaker: str
    start_time: float
    end_time: float


class ExtractEmbeddingRequest(BaseModel):
    audio: SubmitAudio
    segments: List[SpeakerSegment]
    min_duration: Optional[float] = 0.5


@dataclass
class TaskState:
    status_code: str
    created_at: float
    updated_at: float
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    request_params: Optional[SubmitRequest] = None


# In-memory task store & queue (进程内；重启会丢失)
TASKS: Dict[str, TaskState] = {}
TASK_QUEUE: "asyncio.Queue[str]" = asyncio.Queue()

# Lazy-loaded pipelines
_PIPELINE = None
_PIPELINE_MODEL_NAME: Optional[str] = None
_PIPELINE_LOCK = asyncio.Lock()

_DIARIZE_PIPELINE = None
_DIARIZE_MODEL_NAME: Optional[str] = None
_DIARIZE_LOCK = asyncio.Lock()

_EMBEDDING_PIPELINE = None
_EMBEDDING_LOCK = asyncio.Lock()


def _parse_env_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _parse_env_optional_int(value: Optional[str], default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    if v.lower() in {"none", "null"}:
        return None
    return int(v)


def _parse_env_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    return float(v)


def _parse_env_optional_float(value: Optional[str], default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    if v.lower() in {"none", "null"}:
        return None
    return float(v)


def _parse_env_optional_str(value: Optional[str], default: Optional[str]) -> Optional[str]:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return None
    if v.lower() in {"none", "null"}:
        return None
    return v


def _parse_env_int_list(value: Optional[str], default: List[int]) -> List[int]:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    parts = [p.strip() for p in v.split(",") if p.strip() != ""]
    if not parts:
        return default
    return [int(p) for p in parts]


def _build_asr_options_from_env() -> Dict[str, Any]:
    temperature = _parse_env_float(os.environ.get("WHISPERX_TEMPERATURE"), 0.0)
    increment = _parse_env_optional_float(
        os.environ.get("WHISPERX_TEMPERATURE_INCREMENT_ON_FALLBACK"),
        0.2,
    )
    if increment is not None:
        steps = int(round((1.0 - float(temperature)) / float(increment)))
        temperatures = [float(temperature) + (i * float(increment)) for i in range(max(steps, 0) + 1)]
        if temperatures and temperatures[-1] < 1.0:
            temperatures.append(1.0)
        temperatures = [min(max(t, 0.0), 1.0) for t in temperatures]
    else:
        temperatures = [float(temperature)]

    return {
        "beam_size": _parse_env_optional_int(os.environ.get("WHISPERX_BEAM_SIZE"), 5),
        "patience": _parse_env_float(os.environ.get("WHISPERX_PATIENCE"), 1.0),
        "length_penalty": _parse_env_float(os.environ.get("WHISPERX_LENGTH_PENALTY"), 1.0),
        "temperatures": temperatures,
        "compression_ratio_threshold": _parse_env_optional_float(os.environ.get("WHISPERX_COMPRESSION_RATIO_THRESHOLD"), 2.4),
        "log_prob_threshold": _parse_env_optional_float(os.environ.get("WHISPERX_LOGPROB_THRESHOLD"), -1.0),
        "no_speech_threshold": _parse_env_optional_float(os.environ.get("WHISPERX_NO_SPEECH_THRESHOLD"), 0.6),
        "condition_on_previous_text": False,
        "initial_prompt": _parse_env_optional_str(os.environ.get("WHISPERX_INITIAL_PROMPT"), None),
        "hotwords": _parse_env_optional_str(os.environ.get("WHISPERX_HOTWORDS"), None),
        "suppress_tokens": _parse_env_int_list(os.environ.get("WHISPERX_SUPPRESS_TOKENS"), [-1]),
        "suppress_numerals": _parse_env_bool(os.environ.get("WHISPERX_SUPPRESS_NUMERALS"), False),
    }


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def _make_result_payload(audio_duration_ms: int, text: str, utterances: List[Dict[str, Any]]):
    return {
        "audio_info": {"duration": audio_duration_ms},
        "result": {
            "additions": {"duration": audio_duration_ms},
            "text": text,
            "utterances": utterances,
        },
    }


async def _ensure_pipeline(model_name: str):
    """
    初始化/复用 whisperx pipeline。
    """
    global _PIPELINE, _PIPELINE_MODEL_NAME
    async with _PIPELINE_LOCK:
        if _PIPELINE is not None and _PIPELINE_MODEL_NAME == model_name:
            return _PIPELINE

        from whisperx.asr import load_model

        device = "cuda" if os.environ.get("WHISPERX_DEVICE") == "cuda" else "cpu"
        device_index = int(os.environ.get("WHISPERX_DEVICE_INDEX", "0"))
        compute_type = os.environ.get("WHISPERX_COMPUTE_TYPE", "float16" if device == "cuda" else "int8")

        asr_options = _build_asr_options_from_env()

        _PIPELINE = load_model(
            model_name,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            vad_method=os.environ.get("WHISPERX_VAD_METHOD", "pyannote"),
            asr_options=asr_options,
        )
        _PIPELINE_MODEL_NAME = model_name
        return _PIPELINE


async def _ensure_diarize_pipeline(model_name: str, hf_token: str):
    """
    Initialize/Reuse Diarization pipeline.
    """
    global _DIARIZE_PIPELINE, _DIARIZE_MODEL_NAME
    async with _DIARIZE_LOCK:
        if _DIARIZE_PIPELINE is not None and _DIARIZE_MODEL_NAME == model_name:
            return _DIARIZE_PIPELINE

        device = "cuda" if os.environ.get("WHISPERX_DEVICE") == "cuda" else "cpu"
        logger.info(f"Loading diarization model {model_name} on {device}")
        
        _DIARIZE_PIPELINE = DiarizationPipeline(model_name=model_name, use_auth_token=hf_token, device=device)
        _DIARIZE_MODEL_NAME = model_name
        return _DIARIZE_PIPELINE


async def _ensure_embedding_pipeline(hf_token: str):
    global _EMBEDDING_PIPELINE
    async with _EMBEDDING_LOCK:
        if _EMBEDDING_PIPELINE is not None:
            return _EMBEDDING_PIPELINE
        
        device = "cuda" if os.environ.get("WHISPERX_DEVICE") == "cuda" else "cpu"
        _EMBEDDING_PIPELINE = SpeechEmbeddingPipeline(
            model_name="pyannote/wespeaker-voxceleb-resnet34-LM",
            use_auth_token=hf_token,
            device=device
        )
        return _EMBEDDING_PIPELINE


def _run_transcribe_sync(model_name: str, audio_path: str, diarize_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行转写（同步函数）。
    """
    from whisperx.audio import SAMPLE_RATE, load_audio

    audio = load_audio(audio_path)
    duration_ms = _ms(float(audio.shape[0]) / float(SAMPLE_RATE))

    # ASR
    result = _PIPELINE.transcribe(audio, batch_size=8, chunk_size=30, verbose=True)
    segments = result.get("segments") or []
    logger.debug(f"Transcribed segments: {len(segments)}")

    # Diarization
    if diarize_params.get("enable") and _DIARIZE_PIPELINE:
        logger.info("Performing diarization...")
        try:
            min_speakers = diarize_params.get("min_speakers")
            max_speakers = diarize_params.get("max_speakers")
            
            # Pass audio (numpy array) directly to avoid reloading
            diarize_segments, speaker_embeddings = _DIARIZE_PIPELINE(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True
            )
            
            result = assign_word_speakers(diarize_segments, result, speaker_embeddings)
            # Update segments variable as it might be modified in place or we should re-read it
            segments = result.get("segments") or []
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}", exc_info=True)
            # Continue without diarization results
            pass

    full_text = "".join([(seg.get("text") or "") for seg in segments]).strip()
    if not segments or not full_text:
        return _make_result_payload(duration_ms or 1, "no speech detected", [
            {"additions": {"speaker": "1"}, "start_time": 0, "end_time": max(duration_ms, 1), "text": "no speech detected"}
        ])

    utterances: List[Dict[str, Any]] = []
    for seg in segments:
        start_s = float(seg.get("start") or 0.0)
        end_s = float(seg.get("end") or 0.0)
        # Use identified speaker or default to "1"
        speaker = seg.get("speaker") or "1"
        utterances.append({
            "additions": {"speaker": speaker},
            "start_time": _ms(start_s),
            "end_time": _ms(end_s),
            "text": (seg.get("text") or ""),
        })

    payload = _make_result_payload(duration_ms or 1, full_text, utterances)
    
    # Include speaker embeddings if present
    if "speaker_embeddings" in result:
        payload["result"]["speaker_embeddings"] = result["speaker_embeddings"]

    return payload


async def _task_worker():
    """
    后台任务 worker
    """
    loop = asyncio.get_running_loop()
    logger.info("Task worker started")
    while True:
        task_id = await TASK_QUEUE.get()
        state = TASKS.get(task_id)
        if state is None:
            logger.warning(f"Task {task_id} not found in state store")
            TASK_QUEUE.task_done()
            continue

        try:
            logger.info(f"Processing task {task_id}")
            state.status_code = ASR_STATUS_PROCESSING
            state.updated_at = time.time()

            model_name = state.model_name or os.environ.get("WHISPERX_MODEL", "small")
            
            # Prepare Diarization
            request_params = state.request_params
            enable_diarization = request_params.enable_speaker_info if request_params else False
            
            hf_token = os.environ.get("HF_TOKEN")
            if enable_diarization:
                if not hf_token:
                    logger.warning("Diarization requested but HF_TOKEN is missing. Skipping diarization.")
                    enable_diarization = False
                else:
                    await _ensure_diarize_pipeline("pyannote/speaker-diarization-3.1", hf_token)

            tmp_path = getattr(state, "_tmp_audio_path", None)
            if not tmp_path or not os.path.exists(tmp_path):
                raise RuntimeError("audio file missing for task")

            logger.info(f"Ensuring pipeline for model {model_name}")
            await _ensure_pipeline(model_name)
            
            diarize_params = {
                "enable": enable_diarization,
                "min_speakers": request_params.min_speakers if request_params else None,
                "max_speakers": request_params.max_speakers if request_params else None,
            }

            payload = await loop.run_in_executor(
                None,
                _run_transcribe_sync,
                model_name,
                tmp_path,
                diarize_params
            )

            text = ((payload.get("result") or {}).get("text") or "").strip().lower()
            if text == "no speech detected":
                logger.info(f"Task {task_id} finished: No speech detected")
                state.status_code = ASR_STATUS_NO_SPEECH
            else:
                logger.info(f"Task {task_id} finished successfully")
                state.status_code = ASR_STATUS_SUCCESS

            state.result = payload
            state.error_message = None
            state.updated_at = time.time()

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            state.status_code = ASR_STATUS_SYSTEM_ERROR
            state.error_message = str(e)
            state.result = None
            state.updated_at = time.time()
        finally:
            tmp_path = getattr(state, "_tmp_audio_path", None)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if tmp_path:
                try:
                    delattr(state, "_tmp_audio_path")
                except Exception:
                    pass
            TASK_QUEUE.task_done()


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_task_worker())


@app.middleware("http")
async def add_custom_headers(request: Request, call_next):
    log_id = request.headers.get("X-Tt-Logid", str(uuid.uuid4()))
    try:
        response = await call_next(request)

        if "X-Tt-Logid" not in response.headers:
            response.headers["X-Tt-Logid"] = log_id

        if "X-Api-Status-Code" not in response.headers:
            if 200 <= response.status_code < 300:
                response.headers["X-Api-Status-Code"] = ASR_STATUS_SUCCESS
                response.headers["X-Api-Message"] = "OK"
            else:
                response.headers["X-Api-Status-Code"] = "40000000"
                response.headers["X-Api-Message"] = "Request Failed"

        if "X-Api-Message" not in response.headers:
            response.headers["X-Api-Message"] = "OK" if response.headers.get("X-Api-Status-Code") == ASR_STATUS_SUCCESS else "Error"

        return response
    except Exception as e:
        resp = JSONResponse(content={"detail": str(e)}, status_code=500)
        resp.headers["X-Tt-Logid"] = log_id
        resp.headers["X-Api-Status-Code"] = ASR_STATUS_SYSTEM_ERROR
        resp.headers["X-Api-Message"] = str(e)
        return resp


@app.post("/submit")
async def submit_task(
    response: Response,
    payload: TaskSubmission,
    x_api_request_id: Optional[str] = Header(None, alias="X-Api-Request-Id"),
):
    if not payload.audio or not payload.audio.data:
        response.headers["X-Api-Status-Code"] = ASR_STATUS_INVALID_PARAMS
        response.headers["X-Api-Message"] = "Missing audio.data (base64)"
        return {}

    task_id = str(uuid.uuid4())
    now = time.time()
    model_name = "tiny"
    TASKS[task_id] = TaskState(
        status_code=ASR_STATUS_IN_QUEUE,
        created_at=now,
        updated_at=now,
        request_id=x_api_request_id,
        model_name=model_name,
        result=None,
        error_message=None,
        request_params=payload.request,  # Store request params
    )

    try:
        audio_bytes = base64.b64decode(payload.audio.data, validate=False)
    except Exception:
        response.headers["X-Api-Status-Code"] = ASR_STATUS_INVALID_PARAMS
        response.headers["X-Api-Message"] = "Invalid base64 in audio.data"
        TASKS.pop(task_id, None)
        return {}

    fd, tmp_path = tempfile.mkstemp(prefix="whisperx_", suffix=".audio")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
    except Exception as e:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        TASKS.pop(task_id, None)
        response.headers["X-Api-Status-Code"] = ASR_STATUS_SYSTEM_ERROR
        response.headers["X-Api-Message"] = f"Failed to save audio: {e}"
        return {}

    setattr(TASKS[task_id], "_tmp_audio_path", tmp_path)

    response.headers["X-Tt-Logid"] = task_id
    response.headers["X-Api-Status-Code"] = ASR_STATUS_SUCCESS
    response.headers["X-Api-Message"] = "OK"

    await TASK_QUEUE.put(task_id)
    return {}


@app.post("/query")
async def query_task(
    response: Response,
    body: Dict[str, Any] = Body(...),
    x_tt_logid: Optional[str] = Header(None, alias="X-Tt-Logid"),
    x_api_request_id: Optional[str] = Header(None, alias="X-Api-Request-Id"),
):
    del body
    task_id = x_tt_logid or None
    if not task_id:
        task_id = x_api_request_id or None

    if not task_id:
        response.headers["X-Api-Status-Code"] = ASR_STATUS_INVALID_PARAMS
        response.headers["X-Api-Message"] = "Missing X-Tt-Logid"
        return {}

    state = TASKS.get(task_id)
    if state is None:
        response.headers["X-Api-Status-Code"] = ASR_STATUS_INVALID_PARAMS
        response.headers["X-Api-Message"] = "Task Not Found"
        return {}

    response.headers["X-Api-Status-Code"] = state.status_code

    if state.status_code == ASR_STATUS_SUCCESS:
        response.headers["X-Api-Message"] = "OK"
        return state.result or {}
    if state.status_code == ASR_STATUS_NO_SPEECH:
        response.headers["X-Api-Message"] = "No speech detected"
        return state.result or {}
    if state.status_code == ASR_STATUS_PROCESSING:
        response.headers["X-Api-Message"] = "Processing"
        return {}
    if state.status_code == ASR_STATUS_IN_QUEUE:
        response.headers["X-Api-Message"] = "In Queue"
        return {}

    response.headers["X-Api-Message"] = state.error_message or "Failed"
    return {}


@app.post("/extract_embeddings")
async def extract_embeddings(
    response: Response,
    payload: ExtractEmbeddingRequest,
):
    """
    Extract speaker embeddings from audio segments.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        response.headers["X-Api-Status-Code"] = ASR_STATUS_INVALID_PARAMS
        response.headers["X-Api-Message"] = "HF_TOKEN not configured"
        return {}

    if not payload.audio or not payload.audio.data:
        response.headers["X-Api-Status-Code"] = ASR_STATUS_INVALID_PARAMS
        response.headers["X-Api-Message"] = "Missing audio.data (base64)"
        return {}

    # Decode audio
    try:
        audio_bytes = base64.b64decode(payload.audio.data, validate=False)
    except Exception:
        response.headers["X-Api-Status-Code"] = ASR_STATUS_INVALID_PARAMS
        response.headers["X-Api-Message"] = "Invalid base64 in audio.data"
        return {}

    # Save temp file
    fd, tmp_path = tempfile.mkstemp(prefix="whisperx_embed_", suffix=".audio")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        
        # Load audio using whisperx utility (which uses ffmpeg to load and resample)
        # Returns numpy array (channels, samples) or (samples,)
        # load_audio handles resampling to 16000
        audio_np = load_audio(tmp_path)
        
    except Exception as e:
        response.headers["X-Api-Status-Code"] = ASR_STATUS_SYSTEM_ERROR
        response.headers["X-Api-Message"] = f"Failed to process audio: {e}"
        return {}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Ensure pipeline
    try:
        pipeline = await _ensure_embedding_pipeline(hf_token)
    except Exception as e:
        logger.error(f"Failed to load embedding pipeline: {e}", exc_info=True)
        response.headers["X-Api-Status-Code"] = ASR_STATUS_SYSTEM_ERROR
        response.headers["X-Api-Message"] = f"Failed to load model: {e}"
        return {}

    # Identify overlapping regions to exclude
    # A simple timeline mask of overlapping regions
    # Discretize? No, float is fine.
    # Sort segments by start time
    sorted_segments = sorted(payload.segments, key=lambda x: x.start_time)
    
    # Find overlap intervals
    overlaps = []
    if len(sorted_segments) > 1:
        # Check overlaps
        # Basic sweep-line
        events = []
        for s in sorted_segments:
            events.append((s.start_time, 1))
            events.append((s.end_time, -1))
        events.sort(key=lambda x: x[0])
        
        active_count = 0
        start_overlap = None
        
        for t, change in events:
            prev_active = active_count
            active_count += change
            
            if prev_active < 2 and active_count >= 2:
                # Started overlap
                start_overlap = t
            elif prev_active >= 2 and active_count < 2:
                # Ended overlap
                if start_overlap is not None:
                    overlaps.append((start_overlap, t))
                    start_overlap = None

    # Group segments by speaker
    speaker_segments = defaultdict(list)
    for seg in payload.segments:
        # Filter too short segments
        if (seg.end_time - seg.start_time) < (payload.min_duration or 0.5):
            continue
        speaker_segments[seg.speaker].append(seg)

    result_embeddings = {}

    for speaker, segments in speaker_segments.items():
        embeddings_list = []
        
        for seg in segments:
            # Construct time intervals for this segment, subtracting overlaps
            # Valid intervals = [seg.start, seg.end] - overlaps
            # This is 1D boolean logic.
            
            # Start with the full segment
            valid_intervals = [(seg.start_time, seg.end_time)]
            
            for o_start, o_end in overlaps:
                new_intervals = []
                for v_start, v_end in valid_intervals:
                    # No overlap
                    if o_end <= v_start or o_start >= v_end:
                        new_intervals.append((v_start, v_end))
                    else:
                        # Overlap exists
                        # Left part
                        if v_start < o_start:
                            new_intervals.append((v_start, o_start))
                        # Right part
                        if v_end > o_end:
                            new_intervals.append((o_end, v_end))
                valid_intervals = new_intervals
            
            # Process valid intervals
            for v_start, v_end in valid_intervals:
                if (v_end - v_start) < (payload.min_duration or 0.5):
                    continue
                
                # Crop audio
                start_sample = int(v_start * SAMPLE_RATE)
                end_sample = int(v_end * SAMPLE_RATE)
                
                if end_sample > audio_np.shape[0]:
                    end_sample = audio_np.shape[0]
                if start_sample >= end_sample:
                    continue
                    
                crop = audio_np[start_sample:end_sample]
                try:
                    emb = pipeline(crop)
                    # emb shape is (1, dimension) or (dimension,) depending on pyannote version
                    # Usually it's (dimension,) for a single embedding or (1, D)
                    if isinstance(emb, np.ndarray):
                        emb = emb.flatten()
                        embeddings_list.append(emb)
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for speaker {speaker}: {e}")

        if not embeddings_list:
            continue

        # Clustering / Averaging
        embeddings_matrix = np.array(embeddings_list)
        
        # If we have enough samples, use DBSCAN to filter outliers
        # DBSCAN metric='cosine'. eps needs to be small enough (cosine distance)
        # Cosine distance = 1 - cosine similarity. 
        # Same speaker similarity is usually > 0.5 or 0.7. So distance < 0.3 or 0.5.
        if len(embeddings_list) >= 3:
            # eps=0.5 corresponds to similarity 0.5
            clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(embeddings_matrix)
            labels = clustering.labels_
            
            # Find largest cluster (excluding noise -1)
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            
            if len(unique_labels) > 0:
                largest_cluster_label = unique_labels[np.argmax(counts)]
                # Filter embeddings belonging to the largest cluster
                embeddings_matrix = embeddings_matrix[labels == largest_cluster_label]
            else:
                # All noise? Fallback to mean of all
                pass
        
        # Calculate mean embedding (centroid)
        # Normalize first? Pyannote embeddings are usually unit normalized or close to it?
        # Better to just average and then normalize if needed.
        mean_embedding = np.mean(embeddings_matrix, axis=0)
        # result_embeddings[speaker] = mean_embedding.tolist()
        
        # Return as list
        result_embeddings[speaker] = mean_embedding.tolist()

    response.headers["X-Api-Status-Code"] = ASR_STATUS_SUCCESS
    response.headers["X-Api-Message"] = "OK"
    return result_embeddings

@app.get("/")
async def root():
    return {"message": "Server is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
