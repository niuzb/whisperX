import uuid
import os
import json
from typing import Union, List, Optional, Dict, Any
from fastapi import FastAPI, Request, Response, Header, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

class TaskStatus:
    SUCCESS = "20000000"
    PROCESSING = "20000001"
    IN_QUEUE = "20000002"
    SILENT_AUDIO = "20000003"
    INVALID_PARAMS = "45000001"
    EMPTY_AUDIO = "45000002"
    INVALID_AUDIO_FORMAT = "45000151"
    SERVER_BUSY = "55000031"
    
    # Generic codes used in the system
    FAILURE_DEFAULT = "40000000"
    INTERNAL_ERROR = "55000000"

# Directory for storing task files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.join(BASE_DIR, "task_results")
os.makedirs(TASK_DIR, exist_ok=True)

def save_task(task_id: str, data: Dict[str, Any]):
    file_path = os.path.join(TASK_DIR, f"{task_id}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_task(task_id: str) -> Optional[Dict[str, Any]]:
    file_path = os.path.join(TASK_DIR, f"{task_id}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)

# Initialize example task
example_task_id = "67ee89ba-7050-4c04-a3d7-ac61a63499b3"
save_task(example_task_id, {
    "status": TaskStatus.SUCCESS,
    "result": {
        "text": "This is a simulated transcription for the example ID.",
        "utterances": [
            {"text": "This is a simulated", "start_time": 0, "end_time": 1000},
            {"text": "transcription for the example ID.", "start_time": 1000, "end_time": 3000}
        ]
    }
})

class TaskSubmission(BaseModel):
    # 根据实际需求定义请求体
    # Define request body based on actual needs
    audio_url: str = None
    task_type: str = "transcribe"

@app.middleware("http")
async def add_custom_headers(request: Request, call_next):
    # 1. 生成或获取 Log ID
    # Generate or get Log ID
    # 优先使用 Task ID (X-Api-Request-Id) 作为 Log ID
    log_id = request.headers.get("X-Api-Request-Id")
    if not log_id:
        log_id = request.headers.get("X-Tt-Logid", str(uuid.uuid4()))
    
    try:
        response = await call_next(request)
        
        # 2. 设置 X-Tt-Logid
        # Set X-Tt-Logid
        # 如果 endpoint 已经设置了 X-Tt-Logid (例如 submit 生成了新 ID)，就不覆盖
        if "X-Tt-Logid" not in response.headers:
            response.headers["X-Tt-Logid"] = log_id
        
        # 3. 设置状态码和消息
        # Set status code and message
        # 如果 endpoint 已经设置了 Status-Code，就不覆盖
        if "X-Api-Status-Code" not in response.headers:
            # 只有在 HTTP 状态码为 2xx 时才认为是业务成功 (20000000)
            if 200 <= response.status_code < 300:
                response.headers["X-Api-Status-Code"] = TaskStatus.SUCCESS
                response.headers["X-Api-Message"] = "OK"
            else:
                # 其他情况视为失败
                # Consider other cases as failure
                response.headers["X-Api-Status-Code"] = TaskStatus.FAILURE_DEFAULT # 非 200 的默认错误码
                response.headers["X-Api-Message"] = "Request Failed"
        
        # 确保 Message 存在
        if "X-Api-Message" not in response.headers:
             if response.headers.get("X-Api-Status-Code") == TaskStatus.SUCCESS:
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
        response.headers["X-Api-Status-Code"] = TaskStatus.INTERNAL_ERROR # 系统错误码
        response.headers["X-Api-Message"] = str(e)
        
        return response

@app.post("/submit")
async def submit_task(
    response: Response,
    request: Request,
    task: TaskSubmission,
    x_api_request_id: str = Header(None, alias="X-Api-Request-Id")
):
    # 生成新的 Task ID
    task_id =  str(uuid.uuid4())
    
    # 设置 X-Tt-Logid 为 task_id
    response.headers["X-Tt-Logid"] = task_id
    
    # 模拟提交任务的处理
    # Simulate task submission processing
    task_data = {
        "status": TaskStatus.PROCESSING, # 默认立即成功
        "result": {
            "text": "Simulated transcription result.",
            "utterances": []
        }
    }
    save_task(task_id, task_data)
    
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
        response.headers["X-Api-Status-Code"] = TaskStatus.INVALID_PARAMS # 请求参数无效
        response.headers["X-Api-Message"] = "Missing X-Api-Request-Id"
        # 返回空 JSON 或错误信息，根据需求。用户说 "Response Body格式 ：JSON... result ... 仅当识别成功时填写"
        return {}

    task_data = load_task(x_api_request_id)
    
    if not task_data:
        response.headers["X-Api-Status-Code"] = TaskStatus.INVALID_PARAMS # 找不到任务也算参数无效或别的
        response.headers["X-Api-Message"] = "Task Not Found"
        return {}

    status_code = task_data.get("status", TaskStatus.SUCCESS)
    response.headers["X-Api-Status-Code"] = status_code
    
    if status_code == TaskStatus.SUCCESS:
        response.headers["X-Api-Message"] = "OK"
        return {"result": task_data.get("result", {})}
    elif status_code == TaskStatus.PROCESSING:
        response.headers["X-Api-Message"] = "Processing"
        return {}
    elif status_code == TaskStatus.IN_QUEUE:
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
