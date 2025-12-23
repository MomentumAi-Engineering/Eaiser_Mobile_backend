"""
Quick Debug Script to Check Admin Dashboard Data
Run this to see what's in the database
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_admin_reports():
    # Connect to MongoDB
    mongo_uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(
        mongo_uri,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=45000,
    )
    
    db = client[os.getenv("MONGODB_NAME", "eaiser_db_user")]
    collection = db.issues
    
    print("=" * 60)
    print("CHECKING ADMIN DASHBOARD REPORTS")
    print("=" * 60)
    
    # Check total reports
    total = await collection.count_documents({})
    print(f"\n📊 Total Reports in Database: {total}")
    
    # Check reports by status
    statuses = [
        "under_admin_review",
        "needs_review",
        "review_required",
        "pending",
        "screened_out"
    ]
    
    print("\n📋 Reports by Status:")
    for status in statuses:
        count = await collection.count_documents({"status": status})
        if count > 0:
            print(f"  - {status}: {count}")
    
    # Get latest under_admin_review reports
    print("\n🔍 Latest 'under_admin_review' Reports:")
    cursor = collection.find({"status": "under_admin_review"}).sort("created_at", -1).limit(5)
    reports = await cursor.to_list(length=5)
    
    if reports:
        for i, report in enumerate(reports, 1):
            print(f"\n  Report #{i}:")
            print(f"    ID: {report.get('_id')}")
            print(f"    User: {report.get('user_email')}")
            print(f"    Status: {report.get('status')}")
            print(f"    Confidence: {report.get('confidence')}%")
            print(f"    Issue Type: {report.get('issue_type')}")
            print(f"    Created: {report.get('created_at')}")
    else:
        print("  ❌ No reports found with status 'under_admin_review'")
    
    # Check for low confidence reports
    print("\n🔍 Low Confidence Reports (< 70%):")
    cursor = collection.find({"confidence": {"$lt": 70}}).sort("created_at", -1).limit(5)
    low_conf_reports = await cursor.to_list(length=5)
    
    if low_conf_reports:
        for i, report in enumerate(low_conf_reports, 1):
            print(f"\n  Report #{i}:")
            print(f"    ID: {report.get('_id')}")
            print(f"    Status: {report.get('status')}")
            print(f"    Confidence: {report.get('confidence')}%")
    else:
        print("  ❌ No low confidence reports found")
    
    # Check latest 10 reports regardless of status
    print("\n🔍 Latest 10 Reports (All Statuses):")
    cursor = collection.find({}).sort("created_at", -1).limit(10)
    all_reports = await cursor.to_list(length=10)
    
    for i, report in enumerate(all_reports, 1):
        print(f"  {i}. ID: {report.get('_id')} | Status: {report.get('status')} | Conf: {report.get('confidence')}%")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    
    client.close()

if __name__ == "__main__":
    asyncio.run(check_admin_reports())
