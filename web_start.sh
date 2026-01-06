
export WHISPERX_COMPUTE_TYPE=int8

uvicorn whisperx.api_server:app --reload
