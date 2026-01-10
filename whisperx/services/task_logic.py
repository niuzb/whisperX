import asyncio
import os
import tempfile
import numpy as np
from collections import defaultdict
from typing import Any, Dict

from whisperx.log_utils import get_logger
from whisperx.audio import load_audio, SAMPLE_RATE

from .config import (
    ASR_STATUS_SUCCESS,
    ASR_STATUS_PROCESSING,
    ASR_STATUS_INVALID_PARAMS,
    ASR_STATUS_SYSTEM_ERROR,
    ASR_STATUS_NO_SPEECH,
)
from .transport import submit_result, download_audio
from .inference import (
    ensure_pipeline,
    ensure_diarize_pipeline,
    ensure_embedding_pipeline,
    run_transcribe_sync,
    compute_speaker_embedding,
)

logger = get_logger(__name__)


async def process_submit_task(session: Any, task_data: Dict[str, Any]) -> None:
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
                await ensure_diarize_pipeline("pyannote/speaker-diarization-3.1", hf_token)
        
        # 确保 pipeline 已加载
        logger.info(f"Ensuring pipeline for model {model_name}")
        await ensure_pipeline(model_name)
        
        diarize_params = {
            "enable": enable_diarization,
            "min_speakers": request_params.get("min_speakers"),
            "max_speakers": request_params.get("max_speakers"),
        }
        
        # 执行转写
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            None,
            run_transcribe_sync,
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


async def process_embedding_task(session: Any, task_data: Dict[str, Any]) -> None:
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
        pipeline = await ensure_embedding_pipeline(hf_token)
        
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
            
            # Use the helper function from inference.py
            result_embeddings[speaker] = compute_speaker_embedding(embeddings_list)
        
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


async def process_task(session: Any, task_data: Dict[str, Any]) -> None:
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
