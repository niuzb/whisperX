import asyncio
import base64
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI()

from whisperx.log_utils import get_logger

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

    # 以下字段来自客户端，但本服务端暂不强依赖（保留以兼容）
    enable_itn: Optional[bool] = None
    enable_punc: Optional[bool] = None
    enable_speaker_info: Optional[bool] = None
    enable_ddc: Optional[bool] = None
    show_utterances: Optional[bool] = None
    enable_lid: Optional[bool] = None
    context: Optional[str] = None


class TaskSubmission(BaseModel):
    user: Optional[SubmitUser] = None
    audio: SubmitAudio
    request: Optional[SubmitRequest] = None


@dataclass
class TaskState:
    status_code: str
    created_at: float
    updated_at: float
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# In-memory task store & queue (进程内；重启会丢失)
TASKS: Dict[str, TaskState] = {}
TASK_QUEUE: "asyncio.Queue[str]" = asyncio.Queue()

# Lazy-loaded ASR pipeline（避免每个任务都重复加载模型）
_PIPELINE = None
_PIPELINE_MODEL_NAME: Optional[str] = None
_PIPELINE_LOCK = asyncio.Lock()


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
    注意：load_model 会比较重，因此需要锁，且尽量复用。
    """
    global _PIPELINE, _PIPELINE_MODEL_NAME
    async with _PIPELINE_LOCK:
        if _PIPELINE is not None and _PIPELINE_MODEL_NAME == model_name:
            return _PIPELINE

        # 懒加载 whisperx 依赖：避免环境缺推理依赖时 API 服务无法启动
        from whisperx.asr import load_model

        # 默认用 CPU；如果环境有 CUDA 可用，whisperx 内部会用 torch 判断
        device = "cuda" if os.environ.get("WHISPERX_DEVICE") == "cuda" else "cpu"
        device_index = int(os.environ.get("WHISPERX_DEVICE_INDEX", "0"))
        compute_type = os.environ.get("WHISPERX_COMPUTE_TYPE", "float16" if device == "cuda" else "int8")

        asr_options = _build_asr_options_from_env()

        _PIPELINE = load_model(
            model_name,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            # 让 VAD 自己判断语音段；对“无语音”更稳
            vad_method=os.environ.get("WHISPERX_VAD_METHOD", "pyannote"),
            asr_options=asr_options,
        )
        _PIPELINE_MODEL_NAME = model_name
        return _PIPELINE


def _run_transcribe_sync(model_name: str, audio_path: str) -> Dict[str, Any]:
    """
    运行转写（同步函数，便于扔到线程池里跑）。
    返回符合客户端 ASRResponse 的 payload（成功 or 无语音）。
    """
    # 懒加载音频工具（依赖 ffmpeg CLI）
    from whisperx.audio import SAMPLE_RATE, load_audio

    audio = load_audio(audio_path)
    duration_ms = _ms(float(audio.shape[0]) / float(SAMPLE_RATE))

    # whisperx 的 pipeline.transcribe 返回 segments: [{start,end,text}, ...]
    result = _PIPELINE.transcribe(audio, batch_size=8, chunk_size=30, verbose=True)
    segments = result.get("segments") or []
    logger.debug(f"Transcribed segments: {len(segments)}")
    # 简单的“无语音”判断：没有任何段，或拼接文本为空
    full_text = "".join([(seg.get("text") or "") for seg in segments]).strip()
    if not segments or not full_text:
        return _make_result_payload(duration_ms or 1, "no speech detected", [
            {"additions": {"speaker": "1"}, "start_time": 0, "end_time": max(duration_ms, 1), "text": "no speech detected"}
        ])

    utterances: List[Dict[str, Any]] = []
    for seg in segments:
        start_s = float(seg.get("start") or 0.0)
        end_s = float(seg.get("end") or 0.0)
        utterances.append({
            "additions": {"speaker": "1"},
            "start_time": _ms(start_s),
            "end_time": _ms(end_s),
            "text": (seg.get("text") or ""),
        })

    return _make_result_payload(duration_ms or 1, full_text, utterances)


async def _task_worker():
    """
    后台任务 worker：从队列里取 task_id，跑转写，写回 TASKS。
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

            # 决定模型：优先使用 submit 时传入的 request.model_name
            model_name = state.model_name or os.environ.get("WHISPERX_MODEL", "small")

            # 任务里需要 audio_path；我们把它塞到 state.error_message 不合适
            # 这里通过临时文件名约定：task_id -> tmp 文件路径（submit 时会设置）
            tmp_path = getattr(state, "_tmp_audio_path", None)  # type: ignore[attr-defined]
            if not tmp_path or not os.path.exists(tmp_path):
                raise RuntimeError("audio file missing for task")

            # 确保 pipeline 已加载（async）
            logger.info(f"Ensuring pipeline for model {model_name}")
            await _ensure_pipeline(model_name)

            payload = await loop.run_in_executor(
                None,
                _run_transcribe_sync,
                model_name,
                tmp_path,
            )

            # 根据 payload 是否为 no speech，返回不同业务码（客户端支持 20000003）
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
            # 尽量清理临时文件
            tmp_path = getattr(state, "_tmp_audio_path", None)  # type: ignore[attr-defined]
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if tmp_path:
                try:
                    delattr(state, "_tmp_audio_path")  # type: ignore[attr-defined]
                except Exception:
                    pass
            TASK_QUEUE.task_done()


@app.on_event("startup")
async def _startup():
    # 启动一个后台 worker（需要更高吞吐可改多个 worker）
    asyncio.create_task(_task_worker())


@app.middleware("http")
async def add_custom_headers(request: Request, call_next):
    """
    统一补齐客户端依赖的响应头：
    - X-Tt-Logid：如果 handler 已经设置，则不覆盖（submit 必须返回 taskId）
    - X-Api-Status-Code / X-Api-Message：如果 handler 已设置，则不覆盖
    """
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
    """
    提交异步语音转文字任务。
    客户端只关心：
    - 响应头 X-Api-Status-Code == 20000000
    - 响应头 X-Tt-Logid 作为 taskId（后续 /query 带上）
    """
    if not payload.audio or not payload.audio.data:
        response.headers["X-Api-Status-Code"] = ASR_STATUS_INVALID_PARAMS
        response.headers["X-Api-Message"] = "Missing audio.data (base64)"
        return {}

    task_id = str(uuid.uuid4())
    now = time.time()
    # model_name = (payload.request.model_name if payload.request and payload.request.model_name else None)
    model_name = "tiny"
    TASKS[task_id] = TaskState(
        status_code=ASR_STATUS_IN_QUEUE,
        created_at=now,
        updated_at=now,
        request_id=x_api_request_id,
        model_name=model_name,
        result=None,
        error_message=None,
    )

    # decode base64 -> temp file
    try:
        audio_bytes = base64.b64decode(payload.audio.data, validate=False)
    except Exception:
        response.headers["X-Api-Status-Code"] = ASR_STATUS_INVALID_PARAMS
        response.headers["X-Api-Message"] = "Invalid base64 in audio.data"
        # 删除创建的 task
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

    # 把临时路径挂在 state 上（仅内部使用）
    setattr(TASKS[task_id], "_tmp_audio_path", tmp_path)

    # 立刻返回 taskId 到响应头，供客户端轮询
    response.headers["X-Tt-Logid"] = task_id
    response.headers["X-Api-Status-Code"] = ASR_STATUS_SUCCESS
    response.headers["X-Api-Message"] = "OK"

    await TASK_QUEUE.put(task_id)
    return {}


@app.post("/query")
async def query_task(
    response: Response,
    body: Dict[str, Any] = Body(...),  # 客户端会发 '{}'，这里保持兼容
    x_tt_logid: Optional[str] = Header(None, alias="X-Tt-Logid"),
    x_api_request_id: Optional[str] = Header(None, alias="X-Api-Request-Id"),
):
    """
    查询任务状态：
    - 20000000：成功（response body 带完整 ASRResponse）
    - 20000001：处理中（body 空）
    - 20000002：排队中（body 空）
    - 20000003：无语音（body 仍返回 ASRResponse；客户端会用固定 NO_SPEECH）
    其他：失败（body 空 + message）
    """
    del body

    # 客户端按 taskId 传在 X-Tt-Logid
    task_id = x_tt_logid or None
    if not task_id:
        # 兼容：如果没有 X-Tt-Logid，退回用 request_id 作为 key（不推荐）
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
        # 这里返回 payload 也行；客户端会优先用它或用自己的 NO_SPEECH
        return state.result or {}
    if state.status_code == ASR_STATUS_PROCESSING:
        response.headers["X-Api-Message"] = "Processing"
        return {}
    if state.status_code == ASR_STATUS_IN_QUEUE:
        response.headers["X-Api-Message"] = "In Queue"
        return {}

    # 失败
    response.headers["X-Api-Message"] = state.error_message or "Failed"
    return {}


@app.get("/")
async def root():
    return {"message": "WhisperX API Server is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
