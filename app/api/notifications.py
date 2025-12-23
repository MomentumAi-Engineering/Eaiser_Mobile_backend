"""
Notification API Routes for Mobile App
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
import logging

from services.mongodb_optimized_service import get_optimized_mongodb_service
from services.notification_service import (
    get_user_notifications,
    mark_notification_as_read,
    get_unread_count
)

router = APIRouter(
    prefix="/notifications",
    tags=["Notifications"]
)

logger = logging.getLogger(__name__)


@router.get("/user/{user_id}")
async def get_notifications(
    user_id: str,
    unread_only: bool = Query(False, description="Only return unread notifications"),
    limit: int = Query(50, description="Maximum number of notifications")
):
    """
    Get notifications for a user
    
    Args:
        user_id: User's ID
        unread_only: Only return unread notifications
        limit: Maximum notifications to return
    
    Returns:
        List of notifications
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database unavailable")
        
        notifications = await get_user_notifications(
            mongo_service=mongo_service,
            user_id=user_id,
            unread_only=unread_only,
            limit=limit
        )
        
        return {
            "success": True,
            "count": len(notifications),
            "notifications": notifications
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/unread-count")
async def get_unread_notifications_count(user_id: str):
    """
    Get count of unread notifications for a user
    
    Args:
        user_id: User's ID
    
    Returns:
        Count of unread notifications
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database unavailable")
        
        count = await get_unread_count(
            mongo_service=mongo_service,
            user_id=user_id
        )
        
        return {
            "success": True,
            "unread_count": count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching unread count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{notification_id}/mark-read")
async def mark_as_read(
    notification_id: str,
    user_id: str = Query(..., description="User ID for security")
):
    """
    Mark a notification as read
    
    Args:
        notification_id: Notification ID
        user_id: User ID (for security)
    
    Returns:
        Success status
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database unavailable")
        
        success = await mark_notification_as_read(
            mongo_service=mongo_service,
            notification_id=notification_id,
            user_id=user_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found or already read")
        
        return {
            "success": True,
            "message": "Notification marked as read"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        raise HTTPException(status_code=500, detail=str(e))
