
from fastapi import APIRouter
from services.email_service import send_email
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/debug-email")
async def debug_email(to: str):
    logger.info(f"🧪 Debug Email Request to: {to}")
    try:
        success = await send_email(
            to_email=to,
            subject="EAiSER Debug Email",
            html_content="<h1>Debug Email</h1><p>Confirmed delivery capability.</p>",
            text_content="Debug email."
        )
        return {"success": success, "message": f"Email sent to {to}" if success else "Failed to send"}
    except Exception as e:
        logger.error(f"Debug email error: {e}")
        return {"success": False, "error": str(e)}
