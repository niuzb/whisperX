import asyncio
import os
import signal
import sys
import tempfile
from typing import Any, Dict, List, Optional
from collections import defaultdict

import aiohttp

# For Clustering
import numpy as np
from sklearn.cluster import DBSCAN

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

# Configuration
TASK_SERVER_URL = os.environ.get("TASK_SERVER_URL", "http://127.0.0.1:443")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))

# Global flag for graceful shutdown
_shutdown_flag = False

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
            logger.info(f"Diarized segments for task id: {(diarize_segments)}")
            logger.info(f"asr result: {result}")
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
    #print embedding shape
    logger.info(f"speaker_embeddings shape: {result['speaker_embeddings'].shape}")
    return payload


async def fetch_task(session: aiohttp.ClientSession, task_type: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    从服务器获取任务，带重试机制
    """
    url = f"{TASK_SERVER_URL}/v1/task/worker/fetch_task"
    params = {"type": task_type}
    
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to fetch task: HTTP {response.status}, retrying... ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(1)
                        continue
                    logger.warning(f"Failed to fetch task: HTTP {response.status}")
                    return None
                
                data = await response.json()
                if not data.get("success"):
                    # No task available (not an error)
                    return None
                
                task_data = data.get("data", {})
                if not task_data.get("task_id"):
                    return None
                
                return task_data
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout while fetching {task_type} task, retrying... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(1)
                continue
            logger.warning(f"Timeout while fetching {task_type} task after {max_retries} attempts")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error fetching {task_type} task: {e}, retrying... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(1)
                continue
            logger.error(f"Error fetching {task_type} task after {max_retries} attempts: {e}", exc_info=True)
            return None
    
    return None


async def submit_result(session: aiohttp.ClientSession, task_id: str, status_code: str, 
                        result: Optional[Dict[str, Any]] = None, error_message: Optional[str] = None,
                        max_retries: int = 3) -> bool:
    """
    提交任务结果到服务器，带重试机制
    """
    url = f"{TASK_SERVER_URL}/v1/task/worker/submit_result"
    payload = {
        "task_id": task_id,
        "status_code": status_code,
    }
    if result is not None:
        payload["result"] = result
    if error_message is not None:
        payload["error_message"] = error_message
    
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to submit result for task {task_id}: HTTP {response.status}, retrying... ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(1)
                        continue
                    logger.error(f"Failed to submit result for task {task_id}: HTTP {response.status}")
                    return False
                
                data = await response.json()
                return data.get("success", False)
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout while submitting result for task {task_id}, retrying... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(1)
                continue
            logger.error(f"Timeout while submitting result for task {task_id} after {max_retries} attempts")
            return False
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error submitting result for task {task_id}: {e}, retrying... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(1)
                continue
            logger.error(f"Error submitting result for task {task_id} after {max_retries} attempts: {e}", exc_info=True)
            return False
    
    return False


async def download_audio(session: aiohttp.ClientSession, audio_url: str, output_path: str, max_retries: int = 3) -> bool:
    """
    从 URL 下载音频文件到指定路径，带重试机制
    """
    for attempt in range(max_retries):
        try:
            async with session.get(audio_url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status != 200:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to download audio: HTTP {response.status}, retrying... ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(2)
                        continue
                    logger.error(f"Failed to download audio: HTTP {response.status}")
                    return False
                
                with open(output_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                
                return True
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout while downloading audio from {audio_url}, retrying... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(2)
                continue
            logger.error(f"Timeout while downloading audio from {audio_url} after {max_retries} attempts")
            return False
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error downloading audio from {audio_url}: {e}, retrying... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(2)
                continue
            logger.error(f"Error downloading audio from {audio_url} after {max_retries} attempts: {e}", exc_info=True)
            return False
    
    return False


async def process_submit_task(session: aiohttp.ClientSession, task_data: Dict[str, Any]) -> None:
    """
    处理 submit 类型的任务（转写）
    """
    task_id = task_data.get("task_id")
    task_input = task_data.get("input", {})
    audio_url = task_input.get("audio_url")
    
    if not audio_url:
        logger.error(f"Task {task_id}: Missing audio_url in input")
        await submit_result(session, task_id, ASR_STATUS_INVALID_PARAMS, 
                           error_message="Missing audio_url in task input")
        return
    
    logger.info(f"Task {task_id}: Received submit task with task_data: {task_data}")
    # 获取请求参数
    request_params = task_input.get("request", {})
    
    model_name =  os.environ.get("WHISPERX_MODEL", "small")
    enable_diarization = request_params.get("enable_speaker_info", False)
    
    tmp_path = None
    try:
        logger.info(f"Processing submit task {task_id}")
        
        # 下载音频文件
        fd, tmp_path = tempfile.mkstemp(prefix="whisperx_", suffix=".audio")
        os.close(fd)
        
        logger.info(f"Downloading audio from {audio_url}")
        if not await download_audio(session, audio_url, tmp_path):
            raise RuntimeError("Failed to download audio file")
        
        # 准备 Diarization
        hf_token = os.environ.get("HF_TOKEN")
        if enable_diarization:
            if not hf_token:
                logger.warning("Diarization requested but HF_TOKEN is missing. Skipping diarization.")
                enable_diarization = False
            else:
                await _ensure_diarize_pipeline("pyannote/speaker-diarization-3.1", hf_token)
        
        # 确保 pipeline 已加载
        logger.info(f"Ensuring pipeline for model {model_name}")
        await _ensure_pipeline(model_name)
        
        diarize_params = {
            "enable": enable_diarization,
            "min_speakers": request_params.get("min_speakers"),
            "max_speakers": request_params.get("max_speakers"),
        }
        
        # 执行转写
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            None,
            _run_transcribe_sync,
            model_name,
            tmp_path,
            diarize_params
        )
        
        # 判断结果
        text = ((payload.get("result") or {}).get("text") or "").strip().lower()
        if text == "no speech detected":
            logger.info(f"Task {task_id} finished: No speech detected")
            status_code = ASR_STATUS_NO_SPEECH
        else:
            logger.info(f"Task {task_id} finished successfully")
            status_code = ASR_STATUS_SUCCESS
        
        # 提交结果
        await submit_result(session, task_id, status_code, result=payload)
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        await submit_result(session, task_id, ASR_STATUS_SYSTEM_ERROR, 
                           error_message=str(e))
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


async def process_embedding_task(session: aiohttp.ClientSession, task_data: Dict[str, Any]) -> None:
    """
    处理 embedding 类型的任务（提取说话人 embedding）
    """
    task_id = task_data.get("task_id")
    task_input = task_data.get("input", {})
    audio_url = task_input.get("audio_url")
    segments = task_input.get("segments", [])
    min_duration = task_input.get("min_duration", 0.5)
    
    if not audio_url:
        logger.error(f"Task {task_id}: Missing audio_url in input")
        await submit_result(session, task_id, ASR_STATUS_INVALID_PARAMS,
                           error_message="Missing audio_url in task input")
        return
    
    if not segments:
        logger.error(f"Task {task_id}: Missing segments in input")
        await submit_result(session, task_id, ASR_STATUS_INVALID_PARAMS,
                           error_message="Missing segments in task input")
        return
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error(f"Task {task_id}: HF_TOKEN not configured")
        await submit_result(session, task_id, ASR_STATUS_INVALID_PARAMS,
                           error_message="HF_TOKEN not configured")
        return
    
    tmp_path = None
    try:
        logger.info(f"Processing embedding task {task_id}")
        
        # 下载音频文件
        fd, tmp_path = tempfile.mkstemp(prefix="whisperx_embed_", suffix=".audio")
        os.close(fd)
        
        logger.info(f"Downloading audio from {audio_url}")
        if not await download_audio(session, audio_url, tmp_path):
            raise RuntimeError("Failed to download audio file")
        
        # 加载音频
        audio_np = load_audio(tmp_path)
        
        # 确保 embedding pipeline 已加载
        pipeline = await _ensure_embedding_pipeline(hf_token)
        
        # 处理 segments（从 extract_embeddings 逻辑中提取）
        sorted_segments = sorted(segments, key=lambda x: x.get("start_time", 0))
        
        # 查找重叠区间
        overlaps = []
        if len(sorted_segments) > 1:
            events = []
            for s in sorted_segments:
                events.append((s.get("start_time", 0), 1))
                events.append((s.get("end_time", 0), -1))
            events.sort(key=lambda x: x[0])
            
            active_count = 0
            start_overlap = None
            
            for t, change in events:
                prev_active = active_count
                active_count += change
                
                if prev_active < 2 and active_count >= 2:
                    start_overlap = t
                elif prev_active >= 2 and active_count < 2:
                    if start_overlap is not None:
                        overlaps.append((start_overlap, t))
                        start_overlap = None
        
        # 按说话人分组
        speaker_segments = defaultdict(list)
        for seg in segments:
            duration = seg.get("end_time", 0) - seg.get("start_time", 0)
            if duration < min_duration:
                continue
            speaker_segments[seg.get("speaker", "1")].append(seg)
        
        result_embeddings = {}
        
        for speaker, segs in speaker_segments.items():
            embeddings_list = []
            
            for seg in segs:
                valid_intervals = [(seg.get("start_time", 0), seg.get("end_time", 0))]
                
                for o_start, o_end in overlaps:
                    new_intervals = []
                    for v_start, v_end in valid_intervals:
                        if o_end <= v_start or o_start >= v_end:
                            new_intervals.append((v_start, v_end))
                        else:
                            if v_start < o_start:
                                new_intervals.append((v_start, o_start))
                            if v_end > o_end:
                                new_intervals.append((o_end, v_end))
                    valid_intervals = new_intervals
                
                for v_start, v_end in valid_intervals:
                    if (v_end - v_start) < min_duration:
                        continue
                    
                    start_sample = int(v_start * SAMPLE_RATE)
                    end_sample = int(v_end * SAMPLE_RATE)
                    
                    if end_sample > audio_np.shape[0]:
                        end_sample = audio_np.shape[0]
                    if start_sample >= end_sample:
                        continue
                    
                    crop = audio_np[start_sample:end_sample]
                    try:
                        emb = pipeline(crop)
                        if isinstance(emb, np.ndarray):
                            emb = emb.flatten()
                            embeddings_list.append(emb)
                    except Exception as e:
                        logger.warning(f"Failed to extract embedding for speaker {speaker}: {e}")
            
            if not embeddings_list:
                continue
            
            embeddings_matrix = np.array(embeddings_list)
            
            if len(embeddings_list) >= 3:
                clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(embeddings_matrix)
                labels = clustering.labels_
                
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                
                if len(unique_labels) > 0:
                    largest_cluster_label = unique_labels[np.argmax(counts)]
                    embeddings_matrix = embeddings_matrix[labels == largest_cluster_label]
            
            mean_embedding = np.mean(embeddings_matrix, axis=0)
            result_embeddings[speaker] = mean_embedding.tolist()
        
        logger.info(f"Task {task_id} finished successfully")
        await submit_result(session, task_id, ASR_STATUS_SUCCESS, result=result_embeddings)
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        await submit_result(session, task_id, ASR_STATUS_SYSTEM_ERROR,
                           error_message=str(e))
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


async def process_task(session: aiohttp.ClientSession, task_data: Dict[str, Any]) -> None:
    """
    根据任务类型处理任务
    """
    task_type = task_data.get("type", "submit")
    
    if task_type == "submit":
        await process_submit_task(session, task_data)
    elif task_type == "embedding":
        await process_embedding_task(session, task_data)
    else:
        logger.warning(f"Unknown task type: {task_type}")
        task_id = task_data.get("task_id", "unknown")
        await submit_result(session, task_id, ASR_STATUS_INVALID_PARAMS,
                           error_message=f"Unknown task type: {task_type}")


async def main_loop():
    """
    主轮询循环
    """
    global _shutdown_flag
    
    logger.info("Starting WhisperX worker")
    logger.info(f"Server URL: {TASK_SERVER_URL}")
    logger.info(f"Poll interval: {POLL_INTERVAL} seconds")
    
    async with aiohttp.ClientSession() as session:
        while not _shutdown_flag:
            try:
                # 轮询 submit 任务
                submit_task = await fetch_task(session, "submit")
                if submit_task:
                    logger.info(f"Fetched submit task: {submit_task.get('task_id')}")
                    await process_task(session, submit_task)
                    # 处理完任务后立即继续，不等待
                    continue
                
                # 轮询 embedding 任务
                embedding_task = await fetch_task(session, "embedding")
                if embedding_task:
                    logger.info(f"Fetched embedding task: {embedding_task.get('task_id')}")
                    await process_task(session, embedding_task)
                    # 处理完任务后立即继续，不等待
                    continue
                
                # 没有任务，等待一段时间
                await asyncio.sleep(POLL_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                _shutdown_flag = True
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(POLL_INTERVAL)
    
    logger.info("Worker stopped")


def signal_handler(signum, frame):
    """
    信号处理函数，用于优雅退出
    """
    global _shutdown_flag
    logger.info(f"Received signal {signum}, shutting down...")
    _shutdown_flag = True


if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行主循环
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
