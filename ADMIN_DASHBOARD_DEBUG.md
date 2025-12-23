# Admin Dashboard Debug - Quick Fix

## Problem
Reports are in database but not showing in admin dashboard.

## What We Found
✅ Backend is running
✅ Reports are being created (ID: 694a25d9cb77655b79d55e90)
✅ Status is correctly set to "under_admin_review"
✅ Database has 130 total issues
✅ 6 reports are pending review
✅ Debug endpoint shows data exists

## Root Cause
The issue is likely one of these:

1. **Admin Authentication** - Admin not logged in properly
2. **Frontend API Call** - Wrong endpoint or missing auth token
3. **CORS Issue** - Frontend can't access backend
4. **Database Service** - Using wrong MongoDB service (optimized vs regular)

## Quick Fix Steps

### Step 1: Check Admin Login
1. Open admin dashboard: http://localhost:3000/admin/dashboard
2. Check if you're logged in
3. Check browser console for errors (F12)

### Step 2: Test Debug Endpoint
```powershell
# This should return data:
Invoke-WebRequest -Uri "http://localhost:3001/api/admin/review/pending-debug" -UseBasicParsing
```

### Step 3: Check Frontend API Call
Open browser console and check:
- Network tab → Look for `/api/admin/review/pending` call
- Check if it's returning 401 (unauthorized) or 403 (forbidden)
- Check if auth token is being sent

### Step 4: Verify Backend Endpoint
The backend has TWO endpoints:
- `/api/admin/review/pending` - Requires authentication
- `/api/admin/review/pending-debug` - No authentication (for testing)

Frontend might be calling the wrong one or missing auth headers.

## Immediate Solution

### Option A: Use Debug Endpoint Temporarily
Change frontend to call:
```
http://localhost:3001/api/admin/review/pending-debug
```

### Option B: Fix Authentication
1. Login to admin dashboard
2. Check if JWT token is stored in localStorage
3. Verify token is sent in Authorization header

## Files to Check

1. **Frontend API Client:**
   - File: `EAISER_FRONTEND/src/services/apiClient.js`
   - Check if auth token is being sent

2. **Admin Dashboard Component:**
   - Check if it's calling correct endpoint
   - Check if auth headers are included

3. **Backend Admin Review Route:**
   - File: `Eaiser_Mobile_backend/app/routes/admin_review.py`
   - Line 365: `get_pending_reviews` function

## Test Commands

### Test with curl (if available):
```bash
# Without auth (debug endpoint):
curl http://localhost:3001/api/admin/review/pending-debug

# With auth (requires token):
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:3001/api/admin/review/pending
```

### Test with PowerShell:
```powershell
# Debug endpoint:
Invoke-WebRequest -Uri "http://localhost:3001/api/admin/review/pending-debug" -UseBasicParsing

# With auth:
$headers = @{"Authorization" = "Bearer YOUR_TOKEN"}
Invoke-WebRequest -Uri "http://localhost:3001/api/admin/review/pending" -Headers $headers -UseBasicParsing
```

## Expected Response
Should see JSON with reports:
```json
{
  "database": "eaiser_db_user",
  "total_issues": 130,
  "total_pending_review": 6,
  "pending_issues": [
    {
      "id": "694a25d9cb77655b79d55e90",
      "status": "under_admin_review",
      "confidence": 60,
      ...
    }
  ]
}
```

## Next Steps
1. Check browser console for errors
2. Verify admin login status
3. Check if frontend is calling correct endpoint
4. Verify auth token is being sent

---

**Quick Test:** Open http://localhost:3001/api/admin/review/pending-debug in browser
- If you see JSON data → Backend is working, frontend issue
- If you see error → Backend issue
