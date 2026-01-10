import hashlib
import time
import os
import logging
import aiohttp
import asyncio

# Constants
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1450827691828838466/oPFdcZFS7_iG-mFgcfpr1b_esK2hcISDDNXBrZY8xtGzM4MGuXusBtLwtjzAxYOcCKzO'
ALERT_COOLDOWN = 5 * 60  # 5 minutes in seconds

# State
alert_history = {}
logger = logging.getLogger(__name__)

def _clean_history():
    now = time.time()
    to_remove = []
    for key, timestamp in alert_history.items():
        if now - timestamp > ALERT_COOLDOWN:
            to_remove.append(key)
    for key in to_remove:
        del alert_history[key]

async def send_alert(message: str, alert_type: str = 'general', wait: bool = False):
    """
    Sends an alert message to Discord with rate limiting.
    
    Args:
        message: The message content to send.
        alert_type: The type/category of the alert.
        wait: If True, awaits the request. If False, schedules it as a background task.
    """
    if os.environ.get("SKIP_MONITOR_ON_MAC", "false").lower() == "true":
         import platform
         if platform.system() == 'Darwin':
             return

    try:
        # Rate limiting
        _clean_history()
        msg_hash = hashlib.md5(message.encode('utf-8')).hexdigest()
        key = f"{alert_type}:{msg_hash}"
        
        now = time.time()
        last_sent = alert_history.get(key)
        
        if last_sent and (now - last_sent) < ALERT_COOLDOWN:
            return
            
        alert_history[key] = now
        
        coro = _perform_send(message, alert_type)
        
        if wait:
            await coro
        else:
            # Fire and forget
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                logger.warning("send_alert called with wait=False but no running loop. Awaiting explicitly.")
                await coro
        
    except Exception as e:
        logger.error(f"[Alert Service] Error in send_alert: {e}")

async def _perform_send(message: str, alert_type: str):
    payload = {
        "content": f"[{alert_type}] {message}"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(DISCORD_WEBHOOK_URL, json=payload) as response:
                if response.status >= 400:
                    text = await response.text()
                    logger.error(f"Discord API error: {response.status} {text}")
    except Exception as e:
        logger.error(f"[Alert Service] Failed to send alert: {e}")
