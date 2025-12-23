import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_issues():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGODB_NAME", "eaiser_db_user")
    
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    
    # Count by status
    total = await db.issues.count_documents({})
    needs_review = await db.issues.count_documents({"status": "needs_review"})
    under_admin = await db.issues.count_documents({"status": "under_admin_review"})
    pending = await db.issues.count_documents({"status": "pending"})
    analyzing = await db.issues.count_documents({"status": "analyzing"})
    
    print(f"📊 Database: {db_name}")
    print(f"Total Issues: {total}")
    print(f"needs_review: {needs_review}")
    print(f"under_admin_review: {under_admin}")
    print(f"pending: {pending}")
    print(f"analyzing: {analyzing}")
    
    # Get last 5 issues
    print("\n🔍 Last 5 Issues:")
    cursor = db.issues.find({}).sort("timestamp", -1).limit(5)
    async for issue in cursor:
        print(f"  - ID: {str(issue['_id'])[-8:]}")
        print(f"    Status: {issue.get('status', 'N/A')}")
        print(f"    Type: {issue.get('issue_type', 'N/A')}")
        print(f"    Confidence: {issue.get('confidence', 'N/A')}")
        print(f"    Timestamp: {issue.get('timestamp', 'N/A')}")
        print()
    
    client.close()

if __name__ == "__main__":
    asyncio.run(check_issues())
