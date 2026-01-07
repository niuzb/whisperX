
export WHISPERX_COMPUTE_TYPE=int8
export HF_TOKEN=xx
uvicorn whisperx.api_server:app --reload
