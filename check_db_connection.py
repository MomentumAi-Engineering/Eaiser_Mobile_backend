
import asyncio
import os
import sys
from dotenv import load_dotenv
# Add current directory to path
sys.path.append(os.getcwd())
# Also add app directory to path to resolve 'services'
sys.path.append(os.path.join(os.getcwd(), 'app'))

from app.services.mongodb_service import init_db, client

async def test_connection():
    print("⏳ Testing MongoDB Connection...")
    try:
        success = await init_db()
        if success:
            print("\n✅ SUCCESS: Connection Established!")
            print("Your IP is whitelisted and the backend can connect.")
            return True
        else:
            print("\n❌ FAILURE: Could not connect.")
            return False
    except Exception as e:
        print(f"\n❌ FAILURE: Error: {e}")
        return False

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test_connection())
    except KeyboardInterrupt:
        print("\n⚠️ Check cancelled")
