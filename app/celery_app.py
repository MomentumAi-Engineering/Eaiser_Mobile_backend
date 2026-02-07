from celery import Celery
import os
import logging
import asyncio
from typing import Dict, Any, Optional
from bson.objectid import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CeleryWorker")

# Load Env
from dotenv import load_dotenv
load_dotenv()

# Redis URL
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

celery_app = Celery('eaiser_mobile', broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1, # Fair dispatch for heavy AI tasks
    task_acks_late=True,
)

# Async Helper
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@celery_app.task(bind=True, max_retries=3, name="tasks.process_ai_report")
def process_ai_report(self, issue_id: str):
    """
    Background Task: Fetch issue -> Run AI -> Generate Report -> Update DB
    """
    logger.info(f"🤖 Processing AI Report for Issue: {issue_id}")
    
    try:
        # Import Services inside task to avoid import loops / context issues
        from services.mongodb_service import get_db, get_fs
        from services.ai_service import classify_issue, generate_report
        
        async def _process():
            db = await get_db()
            fs = await get_fs()
            
            # 1. Fetch Issue
            try:
                issue = await db.issues.find_one({"_id": issue_id})
                if not issue:
                    issue = await db.issues.find_one({"_id": ObjectId(issue_id)})
            except:
                issue = None
                
            if not issue:
                logger.error(f"❌ Issue {issue_id} not found in DB")
                return "not_found"
                
            # 2. Fetch Image
            image_id = issue.get("image_id")
            if not image_id:
                logger.error("❌ No image_id in issue")
                return "no_image"
                
            image_content = b""
            try:
                grid_out = await fs.open_download_stream(ObjectId(image_id))
                image_content = await grid_out.read()
            except Exception as e:
                logger.error(f"❌ Failed to read image: {e}")
                return "image_error"

            description = issue.get("description", "")
            
            # 3. AI Classification
            issue_type, severity, confidence, category, priority = await classify_issue(image_content, description)
            
            # 4. Generate Full Report
            report_data = await generate_report(
                image_content=image_content,
                description=description,
                issue_type=issue_type,
                severity=severity,
                address=issue.get("address", ""),
                zip_code=issue.get("zip_code", ""),
                latitude=issue.get("latitude", 0.0),
                longitude=issue.get("longitude", 0.0),
                issue_id=issue_id,
                confidence=confidence,
                category=category,
                priority=priority
            )
            
            # 5. Update DB
            update_data = {
                "status": "waiting_review", # Ready for admin
                "report": report_data,
                "ai_processed": True,
                "confidence": confidence,
                "severity": severity,
                "priority": priority,
                # Flatten crucial fields for filtering
                "issue_type": issue_type.title(), 
                "category": category
            }
            
            await db.issues.update_one(
                {"_id": issue["_id"]},
                {"$set": update_data}
            )
            logger.info(f"✅ AI Processing Complete for {issue_id}: Type={issue_type}, Conf={confidence}")
            return "success"

        # Execute
        result = run_async(_process())
        return result

    except Exception as exc:
        logger.error(f"💥 Task Failed: {exc}")
        self.retry(exc=exc, countdown=10)
