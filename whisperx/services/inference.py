import asyncio
import os
import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.cluster import DBSCAN

from whisperx.log_utils import get_logger
from whisperx.diarize import DiarizationPipeline, assign_word_speakers, SpeechEmbeddingPipeline
from whisperx.audio import load_audio, SAMPLE_RATE
from .config import build_asr_options_from_env

logger = get_logger(__name__)

# Lazy-loaded pipelines
_PIPELINE = None
_PIPELINE_MODEL_NAME: Optional[str] = None
_PIPELINE_LOCK = asyncio.Lock()

_DIARIZE_PIPELINE = None
_DIARIZE_MODEL_NAME: Optional[str] = None
_DIARIZE_LOCK = asyncio.Lock()

_EMBEDDING_PIPELINE = None
_EMBEDDING_LOCK = asyncio.Lock()


def ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def make_result_payload(audio_duration_ms: int, text: str, utterances: List[Dict[str, Any]]):
    return {
        "audio_info": {"duration": audio_duration_ms},
        "result": {
            "additions": {"duration": audio_duration_ms},
            "text": text,
            "utterances": utterances,
        },
    }


async def ensure_pipeline(model_name: str):
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

        asr_options = build_asr_options_from_env()

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


async def ensure_diarize_pipeline(model_name: str, hf_token: str):
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


async def ensure_embedding_pipeline(hf_token: str):
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


def run_transcribe_sync(model_name: str, audio_path: str, diarize_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行转写（同步函数）。
    """
    # Note: _PIPELINE and _DIARIZE_PIPELINE are expected to be initialized via ensure_pipeline calls
    # in the asyncio loop before offloading this to a thread.
    
    audio = load_audio(audio_path)
    duration_ms = ms(float(audio.shape[0]) / float(SAMPLE_RATE))

    # ASR
    if _PIPELINE is None:
        raise RuntimeError("ASR Pipeline not initialized")
        
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
        return make_result_payload(duration_ms or 1, "no speech detected", [
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
            "start_time": ms(start_s),
            "end_time": ms(end_s),
            "text": (seg.get("text") or ""),
        })

    payload = make_result_payload(duration_ms or 1, full_text, utterances)
    
    # Include speaker embeddings if present
    if "speaker_embeddings" in result:
        payload["result"]["speaker_embeddings"] = result["speaker_embeddings"]
    #print embedding shape
    if "speaker_embeddings" in result:
        logger.info(f"speaker_embeddings shape: {result['speaker_embeddings'].shape}")
    return payload


def run_diarization_sync(audio_path: str, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run diarization only and return segments.
    """
    if _DIARIZE_PIPELINE is None:
        raise RuntimeError("Diarization Pipeline not initialized")

    audio = load_audio(audio_path)
    
    diarize_segments = _DIARIZE_PIPELINE(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_embeddings=False
    )
    
    results = []
    for _, row in diarize_segments.iterrows():
        results.append({
            "start_time": row['start'],
            "end_time": row['end'],
            "speaker": row['speaker']
        })
    return results


def compute_speaker_embedding(embeddings_list: List[np.ndarray]) -> List[float]:
    """
    Cluster and compute mean embedding for a speaker.
    """
    embeddings_matrix = np.array(embeddings_list)
    
    if len(embeddings_list) >= 3:
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(embeddings_matrix)
        labels = clustering.labels_
        
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        
        if len(unique_labels) > 0:
            largest_cluster_label = unique_labels[np.argmax(counts)]
            embeddings_matrix = embeddings_matrix[labels == largest_cluster_label]
    
    mean_embedding = np.mean(embeddings_matrix, axis=0)
    return mean_embedding.tolist()
