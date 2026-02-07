import sys
from pathlib import Path

# --- PATH SETUP (CRITICAL FOR IMPORTS) ---
# Add 'app' directory to sys.path so 'services' can be imported directly
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import logging
import math
import json
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from bson.objectid import ObjectId

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MobileCore")

# Now imports should work
try:
    from services.mongodb_service import init_db, close_db, get_db, get_fs
    from services.redis_service import init_redis, close_redis
    from routes.auth import router as auth_router
except ImportError as e:
    logger.error(f"❌ Import Error: {e}")
    # Fallback to avoid crash during dev, but functionality will break
    init_db = close_db = get_db = get_fs = None
    init_redis = close_redis = None
    auth_router = None

# --- AI Trigger Helper ---
def trigger_ai_pipeline(issue_id: str):
    """
    Fire-and-forget AI pipeline via Celery.
    """
    try:
        # Import inside function to avoid circular imports
        # Ensure path is correct here too if needed, but module is loaded
        from celery_app import process_ai_report
        # Use delay() for async execution
        process_ai_report.delay(issue_id)
        logger.info(f"🚀 AI Pipeline triggered for {issue_id} (Celery)")
    except ImportError:
        logger.warning(f"⚠️ Celery not found. AI processing skipped for {issue_id}")
    except Exception as e:
        logger.error(f"❌ Failed to trigger AI task: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if init_db:
        logger.info("🧠 Mobile Core (Brain) Starting...")
        await init_db()
        await init_redis()
    yield
    if close_db:
        logger.info("🧠 Mobile Core Shutting Down...")
        await close_db()
        await close_redis()

app = FastAPI(title="Eaiser Mobile Core", lifespan=lifespan)

# --- Mount Auth Routes ---
if auth_router:
    app.include_router(auth_router, prefix="/internal/auth", tags=["Auth"])

# --- Business Logic: Location Verification ---

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # Radius of earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class LocationVerifyRequest(BaseModel):
    user_lat: float
    user_lon: float
    target_lat: float
    target_lon: float

@app.post("/internal/location-verify")
async def verify_location(req: LocationVerifyRequest):
    distance = haversine_distance(req.user_lat, req.user_lon, req.target_lat, req.target_lon)
    threshold = 75.0
    is_valid = distance <= threshold
    logger.info(f"📍 Location Verify: Dist={distance:.2f}m, Threshold={threshold}m, Valid={is_valid}")
    return {
        "verified": is_valid,
        "distance": distance,
        "threshold": threshold,
        "recommend_location": not is_valid
    }

# --- Business Logic: Report Intake ---

@app.post("/internal/report-intake")
async def ingest_report(
    image: UploadFile = File(...),
    description: str = Form(...),
    user_lat: float = Form(...),
    user_lon: float = Form(...),
    timestamp: str = Form(...),
    user_id: str = Form(...),
    metadata: str = Form("{}")
):
    if not get_db:
        raise HTTPException(status_code=503, detail="Database not initialized")
        
    db = await get_db()
    fs = await get_fs()
    
    try:
        # 1. Read and Save Image
        file_content = await image.read()
        filename = f"mobile_{user_id}_{int(datetime.now().timestamp())}.jpg"
        
        image_id = await fs.upload_from_stream(
            filename=filename,
            source=file_content,
            metadata={
                "uploaded_by": user_id,
                "source": "mobile_app",
                "content_type": image.content_type
            }
        )
        
        # 2. Prepare Data
        try:
            meta_dict = json.loads(metadata)
        except:
            meta_dict = {}

        issue_data = {
            "description": description,
            "user_id": user_id,
            "latitude": float(user_lat),
            "longitude": float(user_lon),
            "timestamp": timestamp,
            "created_at": datetime.utcnow(),
            "status": "pending",
            "source": "mobile",
            "image_id": str(image_id),
            "metadata": meta_dict,
            "issue_type": "mobile_report",
            "severity": "medium", 
            "priority": "medium",
            "category": "public"
        }
        
        # 3. Save to DB
        result = await db.issues.insert_one(issue_data)
        
        # ⚡️ TRIGGER AI PIPELINE
        trigger_ai_pipeline(str(result.inserted_id))
        
        logger.info(f"✅ Report saved: {result.inserted_id}")
        
        return {
            "status": "success",
            "report_id": str(result.inserted_id),
            "image_id": str(image_id),
            "verification_status": "pending_ai"
        }
        
    except Exception as e:
        logger.error(f"❌ Report Intake Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Report Status & History ---

@app.get("/internal/report-status/{report_id}")
async def get_report_status(report_id: str):
    if not get_db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    db = await get_db()
    try:
        report = await db.issues.find_one({"_id": report_id}, {"status": 1, "created_at": 1, "decline_reason": 1})
        if not report:
            try:
                report = await db.issues.find_one({"_id": ObjectId(report_id)}, {"status": 1, "created_at": 1, "decline_reason": 1})
            except:
                pass
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "report_id": str(report["_id"]),
            "status": report.get("status", "unknown"),
            "submitted_at": report.get("created_at"),
            "decline_reason": report.get("decline_reason")
        }
    except Exception as e:
        logger.error(f"Error fetching status: {e}")
        raise HTTPException(status_code=500, detail="Internal Error")

@app.get("/internal/user-history/{user_id}")
async def get_user_history(user_id: str):
    if not get_db:
         raise HTTPException(status_code=503, detail="Database not initialized")
    db = await get_db()
    try:
        cursor = db.issues.find({"user_id": user_id}).sort("created_at", -1).limit(50)
        issues = await cursor.to_list(length=50)
        return [
            {
                "id": str(issue["_id"]),
                "description": issue.get("description"),
                "status": issue.get("status"),
                "date": issue.get("created_at"),
                "image_id": issue.get("image_id")
            }
            for issue in issues
        ]
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Internal Error")

# Health Check
@app.get("/")
async def health():
    return {"status": "brain_online", "mode": "hybrid_core"}
