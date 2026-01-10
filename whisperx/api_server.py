import asyncio
import signal
import sys
import aiohttp

from whisperx.log_utils import get_logger
from whisperx.services.config import TASK_SERVER_URL, POLL_INTERVAL
from whisperx.services.transport import fetch_task
from whisperx.services.task_logic import process_task

logger = get_logger(__name__)

# Global flag for graceful shutdown
_shutdown_flag = False


async def main_loop():
    """
    主轮询循环
    """
    global _shutdown_flag
    
    logger.info("Starting WhisperX worker")
    logger.info(f"Server URL: {TASK_SERVER_URL}")
    logger.info(f"Poll interval: {POLL_INTERVAL} seconds")
    
    async with aiohttp.ClientSession() as session:
        while not _shutdown_flag:
            try:
                # 轮询 submit 任务
                submit_task = await fetch_task(session, "submit")
                if submit_task:
                    logger.info(f"Fetched submit task: {submit_task.get('task_id')}")
                    await process_task(session, submit_task)
                    # 处理完任务后立即继续，不等待
                    continue
                
                # 轮询 embedding 任务
                embedding_task = await fetch_task(session, "embedding")
                if embedding_task:
                    logger.info(f"Fetched embedding task: {embedding_task.get('task_id')}")
                    await process_task(session, embedding_task)
                    # 处理完任务后立即继续，不等待
                    continue
                
                # 没有任务，等待一段时间
                await asyncio.sleep(POLL_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                _shutdown_flag = True
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(POLL_INTERVAL)
    
    logger.info("Worker stopped")


def signal_handler(signum, frame):
    """
    信号处理函数，用于优雅退出
    """
    global _shutdown_flag
    logger.info(f"Received signal {signum}, shutting down...")
    _shutdown_flag = True


if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行主循环
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
