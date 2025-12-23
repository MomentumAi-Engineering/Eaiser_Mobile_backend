"""
Enhanced Issues Routes with Confidence-Based Routing
Implements complete workflow: Generate → Review → Confidence Check → Admin Review → Submit
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from services.mongodb_service import get_db, get_fs
from services.ai_service import generate_report, classify_issue
from bson.objectid import ObjectId

logger = logging.getLogger(__name__)
router = APIRouter()

# ============================================================================
# MODELS
# ============================================================================

class AnalyzeImageRequest(BaseModel):
    """Request model for image analysis"""
    pass

class ReviewReportRequest(BaseModel):
    """User reviews and can edit the AI-generated report"""
    edited_summary: Optional[str] = None
    edited_issue_type: Optional[str] = None
    user_notes: Optional[str] = None

class SubmitReportRequest(BaseModel):
    """Submit report after user review"""
    selected_authorities: List[Dict[str, str]]  # [{name, email, type}]
    edited_report: Optional[Dict[str, Any]] = None

class AdminReviewRequest(BaseModel):
    """Admin reviews low-confidence reports"""
    action: str  # "approve" or "reject"
    admin_notes: Optional[str] = None
    edited_report: Optional[Dict[str, Any]] = None

class IssueStatus(BaseModel):
    """Issue status model"""
    issue_id: str
    status: str
    confidence: float
    requires_admin_review: bool
    admin_reviewed: bool = False
    submitted_to_authority: bool = False
    created_at: datetime
    updated_at: datetime

# ============================================================================
# STATUS DEFINITIONS
# ============================================================================

STATUS_ANALYZING = "analyzing"
STATUS_REVIEW_REQUIRED = "review_required"
STATUS_UNDER_ADMIN_REVIEW = "under_admin_review"
STATUS_APPROVED_SUBMITTED = "approved_submitted"
STATUS_SUBMITTED_TO_AUTHORITY = "submitted_to_authority"
STATUS_IN_PROGRESS = "in_progress"
STATUS_RESOLVED = "resolved"
STATUS_REJECTED = "rejected"

CONFIDENCE_THRESHOLD = 70.0  # Reports below this go to admin review

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/issues/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    address: str = Form(...),
    zip_code: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    user_email: str = Form(...)
):
    """
    Step 1: Analyze image with AI and generate report
    Returns report with confidence score for user review
    """
    try:
        logger.info(f"📸 Analyzing image for user: {user_email}")
        
        # Read image
        image_content = await image.read()
        
        # Step 1: Classify the issue using AI
        # Create description from address and zip
        initial_description = f"Issue reported at {address}, ZIP: {zip_code}"
        
        # Classify returns: (issue_type, severity, confidence, category, priority)
        issue_type, severity, confidence, category, priority = await classify_issue(
            image_content=image_content,
            description=initial_description
        )
        
        logger.info(f"🤖 AI Classification: {issue_type} | Severity: {severity} | Confidence: {confidence}%")
        
        # Generate issue ID
        issue_id = str(ObjectId())
        
        # Use initial description for report
        description = f"{issue_type.title()} issue detected at {address}"
        
        # Step 2: Generate full report with all parameters
        report_data = await generate_report(
            image_content=image_content,
            description=description,
            issue_type=issue_type,
            severity=severity,
            address=address,
            zip_code=zip_code,
            latitude=latitude,
            longitude=longitude,
            issue_id=issue_id,
            confidence=confidence,
            category=category,
            priority=priority
        )
        
        # Determine if admin review is required
        requires_admin_review = confidence < CONFIDENCE_THRESHOLD
        
        # Save image to GridFS
        fs = await get_fs()
        file_id = await fs.upload_from_stream(
            image.filename or f"issue_{issue_id}.jpg",
            image_content,
            metadata={"content_type": image.content_type}
        )
        image_id = str(file_id)
        
        # Create issue document
        issue_doc = {
            "_id": issue_id,
            "user_email": user_email,
            "address": address,
            "zip_code": zip_code,
            "latitude": latitude,
            "longitude": longitude,
            "image_filename": image.filename,
            "image_id": image_id,  # ADDED: Store the GridFS ID
            "report": report_data,
            "confidence": confidence,
            "status": STATUS_REVIEW_REQUIRED,
            "requires_admin_review": requires_admin_review,
            "admin_reviewed": False,
            "submitted_to_authority": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "issue_type": issue_type,
            "severity": severity,
            "category": category,
            "description": description
        }
        
        # Store in database
        db = await get_db()
        await db.issues.insert_one(issue_doc)
        
        logger.info(f"✅ Report generated - ID: {issue_id}, Confidence: {confidence}%, Admin Review: {requires_admin_review}")
        
        return {
            "success": True,
            "issue_id": issue_id,
            "report": report_data,
            "confidence": confidence,
            "requires_admin_review": requires_admin_review,
            "status": STATUS_REVIEW_REQUIRED,
            "message": "Report generated successfully. Please review before submitting."
        }
        
    except Exception as e:
        logger.error(f"❌ Error analyzing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze image: {str(e)}")


@router.post("/issues/{issue_id}/review")
async def review_report(issue_id: str, review: ReviewReportRequest):
    """
    Step 2: User reviews the AI-generated report
    Can edit summary, issue type, or add notes
    """
    try:
        db = await get_db()
        issue = await db.issues.find_one({"_id": issue_id})
        
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        # Update with user edits
        update_data = {
            "updated_at": datetime.utcnow()
        }
        
        if review.edited_summary:
            update_data["report.issue_overview.summary_explanation"] = review.edited_summary
        
        if review.edited_issue_type:
            update_data["issue_type"] = review.edited_issue_type
        
        if review.user_notes:
            update_data["user_notes"] = review.user_notes
        
        await db.issues.update_one(
            {"_id": issue_id},
            {"$set": update_data}
        )
        
        logger.info(f"✅ User reviewed report: {issue_id}")
        
        return {
            "success": True,
            "message": "Report updated successfully",
            "issue_id": issue_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error reviewing report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/issues/{issue_id}/submit")
async def submit_report(issue_id: str, submit_data: SubmitReportRequest):
    """
    Step 3: Submit report after user review
    Routes based on confidence:
    - High confidence (>=70%): Direct submit to authorities
    - Low confidence (<70%): Send to admin for review
    """
    try:
        db = await get_db()
        issue = await db.issues.find_one({"_id": issue_id})
        
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        confidence = issue.get("confidence", 0.0)
        requires_admin_review = confidence < CONFIDENCE_THRESHOLD
        
        if requires_admin_review:
            # Low confidence - send to admin review
            await db.issues.update_one(
                {"_id": issue_id},
                {
                    "$set": {
                        "status": STATUS_UNDER_ADMIN_REVIEW,
                        "selected_authorities": submit_data.selected_authorities,
                        "updated_at": datetime.utcnow(),
                        "pending_admin_approval": True
                    }
                }
            )
            
            logger.info(f"📋 Report sent to admin review: {issue_id} (Confidence: {confidence}%)")
            
            return {
                "success": True,
                "status": STATUS_UNDER_ADMIN_REVIEW,
                "message": f"Report confidence is {confidence}%. Sent to admin team for verification.",
                "requires_admin_review": True
            }
        else:
            # High confidence - direct submit to authorities
            # TODO: Implement actual email sending to authorities
            await db.issues.update_one(
                {"_id": issue_id},
                {
                    "$set": {
                        "status": STATUS_SUBMITTED_TO_AUTHORITY,
                        "selected_authorities": submit_data.selected_authorities,
                        "submitted_to_authority": True,
                        "submitted_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"✅ Report submitted to authorities: {issue_id} (Confidence: {confidence}%)")
            
            return {
                "success": True,
                "status": STATUS_SUBMITTED_TO_AUTHORITY,
                "message": "Report submitted to authorities successfully!",
                "requires_admin_review": False
            }
        
    except Exception as e:
        logger.error(f"❌ Error submitting report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/issues/my-reports")
async def get_my_reports(user_email: str, status: Optional[str] = None):
    """
    Get user's reports with real-time status
    Supports filtering by status
    """
    try:
        db = await get_db()
        
        # Build query
        query = {"user_email": user_email}
        if status:
            query["status"] = status
        
        # Get reports
        cursor = db.issues.find(query).sort("created_at", -1).limit(50)
        reports = await cursor.to_list(length=50)
        
        # Convert ObjectId to string
        for report in reports:
            if "_id" in report:
                report["_id"] = str(report["_id"])
        
        logger.info(f"📊 Retrieved {len(reports)} reports for user: {user_email}")
        
        return {
            "success": True,
            "reports": reports,
            "count": len(reports)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting reports: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/stats")
async def get_dashboard_stats(user_email: str):
    """
    Get real-time dashboard statistics for user
    """
    try:
        db = await get_db()
        
        # Count by status
        total = await db.issues.count_documents({"user_email": user_email})
        under_review = await db.issues.count_documents({
            "user_email": user_email,
            "status": STATUS_UNDER_ADMIN_REVIEW
        })
        submitted = await db.issues.count_documents({
            "user_email": user_email,
            "status": {"$in": [STATUS_SUBMITTED_TO_AUTHORITY, STATUS_APPROVED_SUBMITTED]}
        })
        in_progress = await db.issues.count_documents({
            "user_email": user_email,
            "status": STATUS_IN_PROGRESS
        })
        resolved = await db.issues.count_documents({
            "user_email": user_email,
            "status": STATUS_RESOLVED
        })
        pending_review = await db.issues.count_documents({
            "user_email": user_email,
            "status": STATUS_REVIEW_REQUIRED
        })
        
        return {
            "success": True,
            "stats": {
                "total": total,
                "pending_review": pending_review,
                "under_admin_review": under_review,
                "submitted": submitted,
                "in_progress": in_progress,
                "resolved": resolved
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting dashboard stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/review/{issue_id}")
async def admin_review_report(issue_id: str, review: AdminReviewRequest):
    """
    Admin reviews low-confidence reports
    Can approve (submit to authorities) or reject
    """
    try:
        db = await get_db()
        issue = await db.issues.find_one({"_id": issue_id})
        
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        if review.action == "approve":
            # Admin approved - submit to authorities
            await db.issues.update_one(
                {"_id": issue_id},
                {
                    "$set": {
                        "status": STATUS_APPROVED_SUBMITTED,
                        "admin_reviewed": True,
                        "admin_approved": True,
                        "admin_notes": review.admin_notes,
                        "submitted_to_authority": True,
                        "submitted_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"✅ Admin approved report: {issue_id}")
            
            return {
                "success": True,
                "status": STATUS_APPROVED_SUBMITTED,
                "message": "Report approved and submitted to authorities"
            }
        else:
            # Admin rejected
            await db.issues.update_one(
                {"_id": issue_id},
                {
                    "$set": {
                        "status": STATUS_REJECTED,
                        "admin_reviewed": True,
                        "admin_approved": False,
                        "admin_notes": review.admin_notes,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"❌ Admin rejected report: {issue_id}")
            
            return {
                "success": True,
                "status": STATUS_REJECTED,
                "message": "Report rejected by admin"
            }
        
    except Exception as e:
        logger.error(f"❌ Error in admin review: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/issues/{issue_id}/status")
async def get_issue_status(issue_id: str):
    """
    Get current status of an issue with timeline
    """
    try:
        db = await get_db()
        issue = await db.issues.find_one({"_id": issue_id})
        
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        # Build status timeline
        timeline = []
        
        timeline.append({
            "status": STATUS_ANALYZING,
            "timestamp": issue.get("created_at"),
            "description": "AI analyzing image"
        })
        
        timeline.append({
            "status": STATUS_REVIEW_REQUIRED,
            "timestamp": issue.get("created_at"),
            "description": "Report generated - awaiting your review"
        })
        
        if issue.get("status") in [STATUS_UNDER_ADMIN_REVIEW, STATUS_APPROVED_SUBMITTED, STATUS_REJECTED]:
            timeline.append({
                "status": STATUS_UNDER_ADMIN_REVIEW,
                "timestamp": issue.get("updated_at"),
                "description": "Under admin review (low confidence)"
            })
        
        if issue.get("status") in [STATUS_APPROVED_SUBMITTED, STATUS_SUBMITTED_TO_AUTHORITY]:
            timeline.append({
                "status": STATUS_APPROVED_SUBMITTED if issue.get("admin_reviewed") else STATUS_SUBMITTED_TO_AUTHORITY,
                "timestamp": issue.get("submitted_at"),
                "description": "Submitted to authorities"
            })
        
        if issue.get("status") == STATUS_IN_PROGRESS:
            timeline.append({
                "status": STATUS_IN_PROGRESS,
                "timestamp": issue.get("updated_at"),
                "description": "Authority working on issue"
            })
        
        if issue.get("status") == STATUS_RESOLVED:
            timeline.append({
                "status": STATUS_RESOLVED,
                "timestamp": issue.get("resolved_at"),
                "description": "Issue resolved"
            })
        
        return {
            "success": True,
            "issue_id": issue_id,
            "current_status": issue.get("status"),
            "confidence": issue.get("confidence"),
            "requires_admin_review": issue.get("requires_admin_review"),
            "admin_reviewed": issue.get("admin_reviewed", False),
            "timeline": timeline
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting issue status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
