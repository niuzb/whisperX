import sys
import traceback
import asyncio
import logging
from .alert_service import send_alert

logger = logging.getLogger(__name__)

def init_exception_monitor():
    """
    Initializes global exception monitoring.
    Listens for uncaught exceptions.
    """
    sys.excepthook = _handle_uncaught_exception

def _handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    formatted_msg = f"🚨 **Uncaught Exception**\n```\n{error_msg}\n```"
    
    try:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(send_alert(formatted_msg, 'exception_monitor', wait=True))
            else:
                loop.run_until_complete(send_alert(formatted_msg, 'exception_monitor', wait=True))
        except RuntimeError:
            asyncio.run(send_alert(formatted_msg, 'exception_monitor', wait=True))
            
    except Exception as e:
        logger.error(f"Failed to send alert in excepthook: {e}")

async def capture_exception(error: Exception, context: str = '', alert_type: str = 'exception_monitor'):
    """
    Manually captures an exception and sends an alert.
    
    Args:
        error: The exception object.
        context: Optional context description.
        alert_type: The alert type.
    """
    logger.error(f"Captured Exception {context}: {error}", exc_info=error)
    
    tb_str = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    formatted_msg = f"⚠️ **Captured Exception** {context if context else ''}\n```\n{tb_str}\n```"
    
    await send_alert(formatted_msg, alert_type)
