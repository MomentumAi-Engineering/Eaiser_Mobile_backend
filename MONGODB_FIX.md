# MongoDB Connection Issue - Quick Fix Guide

## Problem
The backend is experiencing MongoDB Atlas connection timeouts due to SSL handshake failures. This prevents reports from appearing in the admin dashboard.

## Root Cause
1. **Network/Firewall blocking** MongoDB Atlas connections
2. **Slow internet connection** causing SSL handshake timeouts
3. **MongoDB Atlas IP whitelist** not configured properly

## Quick Fix Options

### Option 1: Add Your IP to MongoDB Atlas Whitelist (RECOMMENDED)
1. Go to https://cloud.mongodb.com/
2. Select your project
3. Click "Network Access" in the left sidebar
4. Click "Add IP Address"
5. Either:
   - Click "Add Current IP Address" (for your current IP)
   - Enter `0.0.0.0/0` (to allow all IPs - for testing only!)
6. Click "Confirm"
7. Wait 1-2 minutes for changes to propagate
8. Restart the backend server

### Option 2: Check Windows Firewall
1. Open Windows Defender Firewall
2. Click "Allow an app through firewall"
3. Find Python and ensure both Private and Public are checked
4. If not listed, click "Allow another app" and add Python

### Option 3: Test MongoDB Connection
Run this command to test connectivity:
```powershell
Test-NetConnection -ComputerName ac-6bsfxzd-shard-00-00.piiox9n.mongodb.net -Port 27017
```

If it fails, your network is blocking MongoDB Atlas.

### Option 4: Use VPN
If your ISP or network is blocking MongoDB Atlas:
1. Connect to a VPN
2. Restart the backend server
3. Try again

### Option 5: Increase Timeouts (Temporary)
Already done in the code:
- `connectTimeoutMS`: 30000ms (30 seconds)
- `socketTimeoutMS`: 45000ms (45 seconds)

## Verification Steps
After fixing:
1. Check backend logs for "✅ MongoDB connected successfully"
2. Login to admin dashboard
3. Check if reports appear in "All Issues" tab
4. Reports with status "under_admin_review" should now be visible

## Current Status
✅ Code is correct - status is set to "under_admin_review"
✅ Admin dashboard filters include "under_admin_review"
❌ MongoDB connection is failing due to network issues
❌ This prevents data from being fetched

## Next Steps
1. Fix MongoDB connection (use Option 1 above)
2. Restart backend: `Ctrl+C` then run again
3. Check admin dashboard - reports should appear
4. If still not working, check browser console for errors
