import asyncio
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'app')) # Add app/ to path
# Also add root just in case
sys.path.append(os.getcwd())

from services.mongodb_service import init_db, get_db
from services.mongodb_optimized_service import get_optimized_mongodb_service

async def create_user():
    print("Connecting to DB...")
    await init_db()
    
    email = "test@eaiser.com"
    password = "password123"
    
    # Try Optimized Service first
    mongo = await get_optimized_mongodb_service()
    if mongo:
        users_col = await mongo.get_collection("users")
        existing = await users_col.find_one({"email": email})
        if not existing:
            await users_col.insert_one({
                "email": email,
                "password": password,
                "name": "Test User",
                "role": "user",
                "created_at": "2024-01-01"
            })
            print(f"✅ User created: {email} / {password}")
        else:
            print(f"User {email} already exists.")
            
    else:
        # Fallback
        db = await get_db()
        existing = await db.users.find_one({"email": email})
        if not existing:
            await db.users.insert_one({
                "email": email,
                "password": password,
                "name": "Test User",
                "role": "user"
            })
            print(f"✅ User created (Fallback DB): {email} / {password}")
        else:
             print(f"User {email} already exists.")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(create_user())
