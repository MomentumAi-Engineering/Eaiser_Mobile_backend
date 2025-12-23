"""
In-App Notification Service for EAiSER Mobile App
Sends notifications to users when admin approves/declines their reports
"""

from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


async def create_notification(
    mongo_service,
    user_id: str,
    title: str,
    message: str,
    notification_type: str,  # 'report_approved', 'report_declined', 'general'
    related_issue_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Create an in-app notification for a user
    
    Args:
        mongo_service: MongoDB service instance
        user_id: User's ID
        title: Notification title
        message: Notification message
        notification_type: Type of notification
        related_issue_id: Related issue/report ID
        metadata: Additional metadata
    
    Returns:
        notification_id: Created notification ID
    """
    try:
        notification_doc = {
            "user_id": user_id,
            "title": title,
            "message": message,
            "type": notification_type,
            "related_issue_id": related_issue_id,
            "metadata": metadata or {},
            "read": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert into notifications collection
        result = await mongo_service.db["notifications"].insert_one(notification_doc)
        notification_id = str(result.inserted_id)
        
        logger.info(f"📬 Notification created for user {user_id}: {title}")
        return notification_id
        
    except Exception as e:
        logger.error(f"❌ Failed to create notification: {e}")
        return None


async def get_user_notifications(
    mongo_service,
    user_id: str,
    unread_only: bool = False,
    limit: int = 50
):
    """
    Get notifications for a user
    
    Args:
        mongo_service: MongoDB service instance
        user_id: User's ID
        unread_only: Only return unread notifications
        limit: Maximum number of notifications to return
    
    Returns:
        List of notifications
    """
    try:
        query = {"user_id": user_id}
        if unread_only:
            query["read"] = False
        
        notifications = await mongo_service.db["notifications"].find(query).sort("created_at", -1).limit(limit).to_list(length=limit)
        
        # Convert ObjectId to string
        for notif in notifications:
            notif["_id"] = str(notif["_id"])
            if "related_issue_id" in notif and notif["related_issue_id"]:
                notif["related_issue_id"] = str(notif["related_issue_id"])
        
        return notifications
        
    except Exception as e:
        logger.error(f"❌ Failed to get notifications: {e}")
        return []


async def mark_notification_as_read(
    mongo_service,
    notification_id: str,
    user_id: str
):
    """
    Mark a notification as read
    
    Args:
        mongo_service: MongoDB service instance
        notification_id: Notification ID
        user_id: User's ID (for security)
    
    Returns:
        bool: Success status
    """
    try:
        from bson.objectid import ObjectId
        
        result = await mongo_service.db["notifications"].update_one(
            {"_id": ObjectId(notification_id), "user_id": user_id},
            {"$set": {"read": True, "updated_at": datetime.utcnow()}}
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Failed to mark notification as read: {e}")
        return False


async def get_unread_count(mongo_service, user_id: str):
    """
    Get count of unread notifications for a user
    
    Args:
        mongo_service: MongoDB service instance
        user_id: User's ID
    
    Returns:
        int: Count of unread notifications
    """
    try:
        count = await mongo_service.db["notifications"].count_documents({
            "user_id": user_id,
            "read": False
        })
        return count
        
    except Exception as e:
        logger.error(f"❌ Failed to get unread count: {e}")
        return 0
