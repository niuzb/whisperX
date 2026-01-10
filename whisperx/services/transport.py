import asyncio
import aiohttp
from typing import Any, Dict, Optional
from whisperx.log_utils import get_logger
from .config import TASK_SERVER_URL

logger = get_logger(__name__)

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
