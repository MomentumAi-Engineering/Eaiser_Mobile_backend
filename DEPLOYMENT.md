
# ⚠️  EAiSER Mobile Backend - Production Deployment ⚠️

# Step 1: Services (Start these first)
# ------------------------------------
# 1. MongoDB Atlas (Cloud) - Ensure IP is whitelisted
# 2. Redis (Cloud/Managed) - Upstash or Elasticache

# Step 2: Environment Variables (Set in Render/Heroku/AWS)
# --------------------------------------------------------
ENV=production
PORT=8001
MONGO_URI=<YOUR_PROD_MONGO_URL>
REDIS_URL=<YOUR_PROD_REDIS_URL>
SECRET_KEY=<SECURE_LONG_RANDOM_STRING>
JWT_SECRET=<SAME_AS_SECRET_KEY>
SENDGRID_API_KEY=<YOUR_SENDGRID_KEY>
GEMINI_API_KEY=<YOUR_GEMINI_KEY>

# Step 3: Deployment (Example: Render)
# ------------------------------------
# A. Python Service (The Brain):
#    - Python 3.11+
#    - Start Command: uvicorn app.mobile_core:app --host 0.0.0.0 --port $PORT --workers 4
#
# B. Celery Worker (The Muscle):
#    - Start Command: celery -A app.celery_app worker --loglevel=info --concurrency=4
#
# C. Go Gateway (The Face) - OPTIONAL for MVP, Mandatory for Scale:
#    - Go 1.20+
#    - Set ENV: PYTHON_CORE_URL=<The URL of Service A>
#    - Start Command: go run gateway/main.go

# Step 4: Maintenance
# -------------------
# - Clean old logs regularly
# - Monitor Redis memory usage
