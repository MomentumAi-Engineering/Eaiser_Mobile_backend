import sys
import os
import logging
import asyncio
import time
from pathlib import Path
from datetime import datetime

# --- ENV & PATH SETUP ---
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / '.env', override=True)

current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# --- FASTAPI IMPORTS ---
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import uvicorn

# --- LOCAL MODULES ---
from services.mongodb_service import init_db, close_db
from services.redis_service import init_redis, close_redis
from services.mongodb_optimized_service import init_optimized_mongodb, close_optimized_mongodb
from services.socket_manager import manager

# --- ROUTES ---
from routes.issues import router as issues_router
from routes.auth import router as auth_router
from routes.admin_review import router as admin_review_router
from api.reports import router as reports_router
from api.ai import router as ai_router

# Enhanced routes with complete workflow
from routes.issues_enhanced import router as issues_enhanced_router
from routes.auth_enhanced import router as auth_enhanced_router

# Notification routes
from api.notifications import router as notifications_router

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- TIMING MIDDLEWARE (Inline for Reliability) ---
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

# --- LIFESPAN CONTEXT MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Eaiser AI Backend (Clean Build v2.0)...")
    try:
        await init_db()
        logger.info("✅ MongoDB service initialized")
    except Exception as e:
        logger.error(f"❌ MongoDB initialization error: {e}", exc_info=True)
    
    try:
        await init_redis()
        logger.info("✅ Redis service initialized")
    except Exception as e:
        logger.error(f"❌ Redis initialization error: {e}", exc_info=True)
    
    try:
        await init_optimized_mongodb()
        logger.info("✅ Optimized MongoDB service initialized")
    except Exception as e:
        logger.error(f"❌ Optimized MongoDB initialization error: {e}", exc_info=True)
    
    logger.info("✅ All services initialized - Server ready!")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down...")
    try:
        await close_db()
        await close_redis()
        await close_optimized_mongodb()
        logger.info("✅ All services closed gracefully")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}", exc_info=True)

# --- APP INITIALIZATION ---
app = FastAPI(title="Eaiser AI Backend v2.0", lifespan=lifespan)

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow ALL for Mobile/Dev simplicity.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TimingMiddleware)

# --- REQUEST LOGGING ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/static"):
        logger.info(f"📥 {request.method} {path}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"💥 Error causing 500: {path} - {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e)})

# --- ROUTER MOUNTING ---
# Enhanced routes FIRST (higher priority)
app.include_router(auth_enhanced_router, prefix="/api", tags=["Auth Enhanced"])
app.include_router(issues_enhanced_router, prefix="/api", tags=["Issues Enhanced"])

# Legacy routes (fallback)
app.include_router(issues_router, prefix="/api", tags=["Issues"])
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(admin_review_router, prefix="/api", tags=["Admin"])
app.include_router(reports_router, prefix="/api/reports", tags=["Reports"])
app.include_router(ai_router, prefix="/api", tags=["AI"])
from api.notifications import router as notifications_router
from routes.debug_email import router as debug_email_router

app.include_router(debug_email_router, prefix="/api", tags=["Debug"])
app.include_router(notifications_router, prefix="/api/notifications", tags=["Notifications"])

# --- WEBSOCKETS ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        manager.disconnect(websocket)

# --- TEST DECLINE ENDPOINT (BYPASS ROUTER) ---
@app.post("/api/test-decline")
async def test_decline(request: Request):
    """
    TEST: Decline endpoint directly in main.py to bypass router issues
    """
    logger.info("🔥🔥🔥 TEST DECLINE HIT IN MAIN.PY!")
    
    try:
        from services.mongodb_optimized_service import get_optimized_mongodb_service
        from datetime import datetime
        import json
        
        # Read body
        raw_body = await request.body()
        payload = json.loads(raw_body.decode('utf-8'))
        logger.info(f"🔥 Payload: {payload}")
        
        issue_id = payload.get("issue_id")
        if not issue_id:
            return JSONResponse(status_code=400, content={"error": "Missing issue_id"})
        
        # Update database
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            return JSONResponse(status_code=503, content={"error": "DB unavailable"})
        
        success = await mongo_service.update_one_optimized(
            collection_name='issues',
            filter_dict={"_id": issue_id},
            update_dict={
                "$set": {
                    "status": "rejected",
                    "admin_review": {
                        "action": "decline",
                        "admin_id": "test",
                        "timestamp": datetime.utcnow(),
                        "notes": payload.get("notes", "Test declined")
                    }
                }
            }
        )
        
        # Try ObjectId
        if not success:
            try:
                from bson.objectid import ObjectId
                success = await mongo_service.update_one_optimized(
                    collection_name='issues',
                    filter_dict={"_id": ObjectId(issue_id)},
                    update_dict={
                        "$set": {
                            "status": "rejected",
                            "admin_review": {
                                "action": "decline",
                                "admin_id": "test",
                                "timestamp": datetime.utcnow(),
                                "notes": payload.get("notes", "Test declined")
                            }
                        }
                    }
                )
            except:
                pass
        
        if not success:
            return JSONResponse(status_code=404, content={"error": "Issue not found"})
        
        # Send IN-APP notification to user (instead of email)
        try:
            from services.notification_service import create_notification
            mongo_service_read = await get_optimized_mongodb_service()
            issue_data = await mongo_service_read.get_issue_by_id(issue_id)
            
            if issue_data and issue_data.get("user_id"):
                user_id = issue_data["user_id"]
                decline_reason = payload.get("notes", "Your report did not meet our guidelines")
                
                # Create in-app notification
                await create_notification(
                    mongo_service=mongo_service_read,
                    user_id=user_id,
                    title="❌ Report Declined",
                    message=f"Your report has been reviewed and declined. Reason: {decline_reason}",
                    notification_type="report_declined",
                    related_issue_id=issue_id,
                    metadata={
                        "decline_reason": decline_reason,
                        "admin_action": "decline",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                logger.info(f"📬 In-app notification created for user {user_id}")
        except Exception as notif_error:
            logger.error(f"⚠️ Failed to create notification: {notif_error}")
            # Don't fail the decline operation if notification fails
        
        logger.info(f"🔥🔥🔥 TEST DECLINE SUCCESS!")
        return JSONResponse(status_code=200, content={"message": "Test decline success", "issue_id": issue_id})
    
    except Exception as e:
        logger.info(f"🔥 Test decline error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- TEST APPROVE ENDPOINT ---
@app.post("/api/test-approve")
async def test_approve(request: Request):
    """
    TEST: Approve endpoint with email notification
    """
    logger.info("✅ TEST APPROVE HIT IN MAIN.PY!")
    
    try:
        from services.mongodb_optimized_service import get_optimized_mongodb_service
        from datetime import datetime
        import json
        
        # Read body
        raw_body = await request.body()
        payload = json.loads(raw_body.decode('utf-8'))
        logger.info(f"✅ Payload: {payload}")
        
        issue_id = payload.get("issue_id")
        if not issue_id:
            return JSONResponse(status_code=400, content={"error": "Missing issue_id"})
        
        # Update database
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            return JSONResponse(status_code=503, content={"error": "DB unavailable"})
        
        success = await mongo_service.update_one_optimized(
            collection_name='issues',
            filter_dict={"_id": issue_id},
            update_dict={
                "$set": {
                    "status": "approved",
                    "admin_review": {
                        "action": "approve",
                        "admin_id": "test",
                        "timestamp": datetime.utcnow(),
                        "notes": payload.get("notes", "Approved by admin")
                    }
                }
            }
        )
        
        # Try ObjectId
        if not success:
            try:
                from bson.objectid import ObjectId
                success = await mongo_service.update_one_optimized(
                    collection_name='issues',
                    filter_dict={"_id": ObjectId(issue_id)},
                    update_dict={
                        "$set": {
                            "status": "approved",
                            "admin_review": {
                                "action": "approve",
                                "admin_id": "test",
                                "timestamp": datetime.utcnow(),
                                "notes": payload.get("notes", "Approved by admin")
                            }
                        }
                    }
                )
            except:
                pass
        
        if not success:
            return JSONResponse(status_code=404, content={"error": "Issue not found"})
        
        # Send IN-APP notification to user
        try:
            from services.notification_service import create_notification
            mongo_service_read = await get_optimized_mongodb_service()
            issue_data = await mongo_service_read.get_issue_by_id(issue_id)
            
            if issue_data and issue_data.get("user_id"):
                user_id = issue_data["user_id"]
                
                # Create in-app notification
                await create_notification(
                    mongo_service=mongo_service_read,
                    user_id=user_id,
                    title="✅ Report Approved",
                    message="Great news! Your report has been approved and forwarded to the authorities.",
                    notification_type="report_approved",
                    related_issue_id=issue_id,
                    metadata={
                        "admin_action": "approve",
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "approved"
                    }
                )
                logger.info(f"📬 Approval notification created for user {user_id}")
            
            # --- EMAIL TRIGGER (Testing) ---
            if issue_data and issue_data.get("report"):
                try:
                    from routes.issues import send_authority_email
                    # Need image content
                    from services.mongodb_service import get_fs
                    fs = await get_fs()
                    image_id = issue_data.get("image_id")
                    image_content = b""
                    if image_id:
                        try:
                             from bson.objectid import ObjectId
                             grid_out = await fs.open_download_stream(ObjectId(image_id))
                             image_content = await grid_out.read()
                        except Exception: 
                             pass
                    
                    # Prepare authorities
                    # Minimal mock for testing if missing in report
                    authorities = issue_data["report"].get("responsible_authorities_or_parties", [])
                    if not authorities:
                        authorities = [{"name": "Test Authority", "email":"eaiser@momntumai.com", "type":"general"}]

                    email_success = await send_authority_email(
                        issue_id=str(issue_id),
                        authorities=authorities,
                        issue_type=issue_data.get("issue_type", "Test Issue"),
                        final_address=issue_data.get("address", "Test Address"),
                        zip_code=issue_data.get("zip_code", "N/A"),
                        timestamp_formatted=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                        report=issue_data["report"],
                        confidence=float(issue_data.get("confidence", 0)),
                        category=issue_data.get("category", "public"),
                        timezone_name="UTC",
                        latitude=float(issue_data.get("latitude", 0.0)),
                        longitude=float(issue_data.get("longitude", 0.0)),
                        image_content=image_content,
                        is_user_review=False
                    )
                    logger.info(f"📧 Test Email Trigger Result: {email_success}")
                except Exception as email_err:
                    logger.error(f"❌ Test Email Trigger Failed: {email_err}", exc_info=True)

        except Exception as notif_error:
            logger.error(f"⚠️ Failed to create approval notification: {notif_error}")
            # Don't fail the approve operation if notification fails
        
        logger.info(f"✅✅✅ TEST APPROVE SUCCESS!")
        return JSONResponse(status_code=200, content={"message": "Test approve success", "issue_id": issue_id})
    
    except Exception as e:
        logger.error(f"✅ Test approve error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- HEALTH ---
@app.get("/")
async def root():
    return {"status": "online", "version": "2.0.0", "timestamp": datetime.now().isoformat()}

# --- MAIN ---
if __name__ == "__main__":
    # Force Port 3001
    port = 3001
    logger.info(f"Server starting on port {port}")
    # Use string reference for proper reload support
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
