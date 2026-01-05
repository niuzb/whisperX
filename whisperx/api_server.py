import uuid
from typing import Union
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

class TaskSubmission(BaseModel):
    # 根据实际需求定义请求体
    # Define request body based on actual needs
    audio_url: str = None
    task_type: str = "transcribe"

@app.middleware("http")
async def add_custom_headers(request: Request, call_next):
    # 1. 生成或获取 Log ID
    # Generate or get Log ID
    log_id = request.headers.get("X-Tt-Logid", str(uuid.uuid4()))
    
    try:
        response = await call_next(request)
        
        # 2. 设置 X-Tt-Logid
        # Set X-Tt-Logid
        response.headers["X-Tt-Logid"] = log_id
        
        # 3. 设置状态码和消息
        # Set status code and message
        # 只有在 HTTP 状态码为 2xx 时才认为是业务成功 (20000000)
        if 200 <= response.status_code < 300:
            response.headers["X-Api-Status-Code"] = "20000000"
            response.headers["X-Api-Message"] = "OK"
        else:
            # 其他情况视为失败
            # Consider other cases as failure
            response.headers["X-Api-Status-Code"] = "40000000" # 非 200 的默认错误码
            # 如果 response 已经有了 content，这里 header 的 message 最好简单一点
            response.headers["X-Api-Message"] = "Request Failed"
            
        return response
        
    except Exception as e:
        # 捕获未处理的异常
        # Catch unhandled exceptions
        content = {"detail": str(e)}
        response = JSONResponse(content=content, status_code=500)
        
        response.headers["X-Tt-Logid"] = log_id
        response.headers["X-Api-Status-Code"] = "50000000" # 系统错误码
        response.headers["X-Api-Message"] = str(e)
        
        return response

@app.post("/submit")
async def submit_task(task: TaskSubmission):
    # 模拟提交任务的处理
    # Simulate task submission processing
    return {"status": "success", "task_id": str(uuid.uuid4())}

@app.get("/")
async def root():
    return {"message": "WhisperX API Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
