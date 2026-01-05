import uuid
from typing import Union, List, Optional, Dict, Any
from fastapi import FastAPI, Request, Response, Header, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# In-memory storage for tasks (simulated database)
TASKS: Dict[str, Dict[str, Any]] = {
    "67ee89ba-7050-4c04-a3d7-ac61a63499b3": {
        "status": "20000000",
        "result": {
            "text": "This is a simulated transcription for the example ID.",
            "utterances": [
                {"text": "This is a simulated", "start_time": 0, "end_time": 1000},
                {"text": "transcription for the example ID.", "start_time": 1000, "end_time": 3000}
            ]
        }
    }
}

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
        # 如果 endpoint 已经设置了 Status-Code，就不覆盖
        if "X-Api-Status-Code" not in response.headers:
            # 只有在 HTTP 状态码为 2xx 时才认为是业务成功 (20000000)
            if 200 <= response.status_code < 300:
                response.headers["X-Api-Status-Code"] = "20000000"
                response.headers["X-Api-Message"] = "OK"
            else:
                # 其他情况视为失败
                # Consider other cases as failure
                response.headers["X-Api-Status-Code"] = "40000000" # 非 200 的默认错误码
                response.headers["X-Api-Message"] = "Request Failed"
        
        # 确保 Message 存在
        if "X-Api-Message" not in response.headers:
             if response.headers.get("X-Api-Status-Code") == "20000000":
                 response.headers["X-Api-Message"] = "OK"
             else:
                 response.headers["X-Api-Message"] = "Error"
            
        return response
        
    except Exception as e:
        # 捕获未处理的异常
        # Catch unhandled exceptions
        content = {"detail": str(e)}
        response = JSONResponse(content=content, status_code=500)
        
        response.headers["X-Tt-Logid"] = log_id
        response.headers["X-Api-Status-Code"] = "55000000" # 系统错误码
        response.headers["X-Api-Message"] = str(e)
        
        return response

@app.post("/submit")
async def submit_task(
    request: Request,
    task: TaskSubmission,
    x_api_request_id: str = Header(None, alias="X-Api-Request-Id")
):
    # 使用 header 中的 ID 或者生成新的
    task_id = x_api_request_id if x_api_request_id else str(uuid.uuid4())
    
    # 模拟提交任务的处理
    # Simulate task submission processing
    TASKS[task_id] = {
        "status": "20000000", # 默认立即成功
        "result": {
            "text": "Simulated transcription result.",
            "utterances": []
        }
    }
    
    return {"status": "success", "task_id": task_id}

@app.post("/query")
async def query_task(
    response: Response,
    body: dict = Body(...), # 接收空 JSON
    x_api_request_id: str = Header(None, alias="X-Api-Request-Id"),
    x_api_resource_id: str = Header(None, alias="X-Api-Resource-Id"),
    x_api_app_key: str = Header(None, alias="X-Api-App-Key"),
    x_api_access_key: str = Header(None, alias="X-Api-Access-Key")
):
    if not x_api_request_id:
        response.headers["X-Api-Status-Code"] = "45000001" # 请求参数无效
        response.headers["X-Api-Message"] = "Missing X-Api-Request-Id"
        # 返回空 JSON 或错误信息，根据需求。用户说 "Response Body格式 ：JSON... result ... 仅当识别成功时填写"
        return {}

    task_data = TASKS.get(x_api_request_id)
    
    if not task_data:
        response.headers["X-Api-Status-Code"] = "45000001" # 找不到任务也算参数无效或别的
        response.headers["X-Api-Message"] = "Task Not Found"
        return {}

    status_code = task_data.get("status", "20000000")
    response.headers["X-Api-Status-Code"] = status_code
    
    if status_code == "20000000":
        response.headers["X-Api-Message"] = "OK"
        return {"result": task_data.get("result", {})}
    elif status_code == "20000001":
        response.headers["X-Api-Message"] = "Processing"
        return {}
    elif status_code == "20000002":
        response.headers["X-Api-Message"] = "In Queue"
        return {}
    else:
        response.headers["X-Api-Message"] = "Failed"
        return {}

@app.get("/")
async def root():
    return {"message": "WhisperX API Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
