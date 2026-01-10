import os
from typing import Any, Dict, List, Optional

# Constants
ASR_STATUS_SUCCESS = "20000000"
ASR_STATUS_PROCESSING = "20000001"
ASR_STATUS_IN_QUEUE = "20000002"
ASR_STATUS_NO_SPEECH = "20000003"

ASR_STATUS_INVALID_PARAMS = "45000001"
ASR_STATUS_SYSTEM_ERROR = "55000000"

# Configuration
TASK_SERVER_URL = os.environ.get("TASK_SERVER_URL", "http://127.0.0.1:443")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))


def parse_env_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def parse_env_optional_int(value: Optional[str], default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    if v.lower() in {"none", "null"}:
        return None
    return int(v)


def parse_env_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    return float(v)


def parse_env_optional_float(value: Optional[str], default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    if v.lower() in {"none", "null"}:
        return None
    return float(v)


def parse_env_optional_str(value: Optional[str], default: Optional[str]) -> Optional[str]:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return None
    if v.lower() in {"none", "null"}:
        return None
    return v


def parse_env_int_list(value: Optional[str], default: List[int]) -> List[int]:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    parts = [p.strip() for p in v.split(",") if p.strip() != ""]
    if not parts:
        return default
    return [int(p) for p in parts]


def build_asr_options_from_env() -> Dict[str, Any]:
    temperature = parse_env_float(os.environ.get("WHISPERX_TEMPERATURE"), 0.0)
    increment = parse_env_optional_float(
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
        "beam_size": parse_env_optional_int(os.environ.get("WHISPERX_BEAM_SIZE"), 5),
        "patience": parse_env_float(os.environ.get("WHISPERX_PATIENCE"), 1.0),
        "length_penalty": parse_env_float(os.environ.get("WHISPERX_LENGTH_PENALTY"), 1.0),
        "temperatures": temperatures,
        "compression_ratio_threshold": parse_env_optional_float(os.environ.get("WHISPERX_COMPRESSION_RATIO_THRESHOLD"), 2.4),
        "log_prob_threshold": parse_env_optional_float(os.environ.get("WHISPERX_LOGPROB_THRESHOLD"), -1.0),
        "no_speech_threshold": parse_env_optional_float(os.environ.get("WHISPERX_NO_SPEECH_THRESHOLD"), 0.6),
        "condition_on_previous_text": False,
        "initial_prompt": parse_env_optional_str(os.environ.get("WHISPERX_INITIAL_PROMPT"), None),
        "hotwords": parse_env_optional_str(os.environ.get("WHISPERX_HOTWORDS"), None),
        "suppress_tokens": parse_env_int_list(os.environ.get("WHISPERX_SUPPRESS_TOKENS"), [-1]),
        "suppress_numerals": parse_env_bool(os.environ.get("WHISPERX_SUPPRESS_NUMERALS"), False),
    }
