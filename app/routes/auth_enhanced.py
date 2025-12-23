"""
Advanced Authentication System for Mobile App
Features: Email/Password, OTP, Biometric, Session Management, Password Reset
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime, timedelta
import logging
import secrets
import hashlib
import jwt
from services.mongodb_service import get_db
from services.email_service import send_email
import os

logger = logging.getLogger(__name__)
router = APIRouter()

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = 30
OTP_EXPIRATION_MINUTES = 10

# ============================================================================
# MODELS
# ============================================================================

class SignupInitRequest(BaseModel):
    email: EmailStr
    name: str

class SignupVerifyRequest(BaseModel):
    email: EmailStr
    name: str
    otp: str
    password: str
    device_id: Optional[str] = None
    device_name: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    device_id: Optional[str] = None
    remember_me: bool = False

class BiometricSetupRequest(BaseModel):
    email: EmailStr
    biometric_token: str  # Encrypted biometric data from device
    device_id: str

class BiometricLoginRequest(BaseModel):
    email: EmailStr
    biometric_token: str
    device_id: str

class PasswordResetInitRequest(BaseModel):
    email: EmailStr

class PasswordResetVerifyRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_otp() -> str:
    """Generate 6-digit OTP"""
    return str(secrets.randbelow(900000) + 100000)

def create_access_token(user_id: str, email: str, remember_me: bool = False) -> dict:
    """Create JWT access token and refresh token"""
    expiration = datetime.utcnow() + timedelta(days=JWT_EXPIRATION_DAYS if remember_me else 1)
    
    access_token_data = {
        "user_id": user_id,
        "email": email,
        "exp": expiration,
        "iat": datetime.utcnow()
    }
    
    access_token = jwt.encode(access_token_data, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    # Refresh token (longer expiration)
    refresh_expiration = datetime.utcnow() + timedelta(days=90)
    refresh_token_data = {
        "user_id": user_id,
        "email": email,
        "exp": refresh_expiration,
        "type": "refresh"
    }
    
    refresh_token = jwt.encode(refresh_token_data, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expiration.isoformat(),
        "token_type": "Bearer"
    }

def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(authorization: str = Header(None)):
    """Dependency to get current user from token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.split(" ")[1]
    payload = verify_token(token)
    
    return payload

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/auth/signup-init")
async def signup_init(request: SignupInitRequest):
    """
    Step 1 of signup: Send OTP to email
    """
    try:
        db = await get_db()
        
        # Check if user already exists
        existing_user = await db.users.find_one({"email": request.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Generate OTP
        otp = generate_otp()
        otp_expiry = datetime.utcnow() + timedelta(minutes=OTP_EXPIRATION_MINUTES)
        
        # Store OTP in pending_users collection
        await db.pending_users.update_one(
            {"email": request.email},
            {
                "$set": {
                    "email": request.email,
                    "name": request.name,
                    "otp": otp,
                    "otp_expiry": otp_expiry,
                    "created_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        
        # Send OTP email
        email_subject = "Your EAiSER Verification Code"
        email_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background: #f4f4f4; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 10px;">
                <h1 style="color: #f6c521;">Welcome to EAiSER!</h1>
                <p>Hi {request.name},</p>
                <p>Your verification code is:</p>
                <div style="background: #f6c521; color: white; font-size: 32px; font-weight: bold; text-align: center; padding: 20px; border-radius: 8px; letter-spacing: 8px;">
                    {otp}
                </div>
                <p style="color: #666; margin-top: 20px;">This code will expire in {OTP_EXPIRATION_MINUTES} minutes.</p>
                <p style="color: #666;">If you didn't request this code, please ignore this email.</p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
                <p style="color: #999; font-size: 12px;">© 2025 EAiSER - Civic Reporting Platform</p>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Welcome to EAiSER!
        
        Hi {request.name},
        
        Your verification code is: {otp}
        
        This code will expire in {OTP_EXPIRATION_MINUTES} minutes.
        
        If you didn't request this code, please ignore this email.
        """
        
        await send_email(
            to_email=request.email,
            subject=email_subject,
            html_content=email_body,
            text_content=text_content
        )
        
        logger.info(f"✅ OTP sent to {request.email}")
        
        return {
            "success": True,
            "message": "Verification code sent to your email",
            "email": request.email
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"❌ Signup init error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/signup-verify")
async def signup_verify(request: SignupVerifyRequest):
    """
    Step 2 of signup: Verify OTP and create account
    """
    try:
        db = await get_db()
        
        # Get pending user
        pending_user = await db.pending_users.find_one({"email": request.email})
        if not pending_user:
            raise HTTPException(status_code=404, detail="No pending signup found for this email")
        
        # Verify OTP
        if pending_user["otp"] != request.otp:
            raise HTTPException(status_code=400, detail="Invalid verification code")
        
        # Check OTP expiry
        if datetime.utcnow() > pending_user["otp_expiry"]:
            raise HTTPException(status_code=400, detail="Verification code expired")
        
        # Create user account
        user_doc = {
            "email": request.email,
            "name": request.name,  # Use name from request
            "password_hash": hash_password(request.password),
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow(),
            "email_verified": True,
            "biometric_enabled": False,
            "devices": []
        }
        
        # Add device if provided
        if request.device_id:
            user_doc["devices"] = [{
                "device_id": request.device_id,
                "device_name": request.device_name or "Unknown Device",
                "added_at": datetime.utcnow(),
                "last_used": datetime.utcnow()
            }]
        
        result = await db.users.insert_one(user_doc)
        user_id = str(result.inserted_id)
        
        # Delete pending user
        await db.pending_users.delete_one({"email": request.email})
        
        # Generate tokens
        tokens = create_access_token(user_id, request.email, remember_me=True)
        
        logger.info(f"✅ User account created: {request.email}")
        
        return {
            "success": True,
            "message": "Account created successfully",
            "user": {
                "id": user_id,
                "email": request.email,
                "name": request.name
            },
            **tokens
        }
        
    except Exception as e:
        logger.error(f"❌ Signup verify error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/login")
async def login(request: LoginRequest):
    """
    Login with email and password
    """
    try:
        db = await get_db()
        
        # Find user
        user = await db.users.find_one({"email": request.email})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Verify password (handle both old and new format)
        stored_password = user.get("password_hash") or user.get("password")
        if not stored_password:
            raise HTTPException(status_code=401, detail="Invalid email or password")
            
        if stored_password != hash_password(request.password):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Update last login
        await db.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "last_login": datetime.utcnow()
                }
            }
        )
        
        # Update device info if provided
        if request.device_id:
            await db.users.update_one(
                {"_id": user["_id"], "devices.device_id": request.device_id},
                {
                    "$set": {
                        "devices.$.last_used": datetime.utcnow()
                    }
                }
            )
        
        # Generate tokens
        user_id = str(user["_id"])
        tokens = create_access_token(user_id, request.email, remember_me=request.remember_me)
        
        logger.info(f"✅ User logged in: {request.email}")
        
        return {
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user_id,
                "email": user["email"],
                "name": user["name"],
                "biometric_enabled": user.get("biometric_enabled", False)
            },
            **tokens
        }
        
    except Exception as e:
        logger.error(f"❌ Login error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/biometric/setup")
async def setup_biometric(request: BiometricSetupRequest, current_user: dict = Depends(get_current_user)):
    """
    Enable biometric authentication for a device
    """
    try:
        db = await get_db()
        
        # Update user with biometric token
        await db.users.update_one(
            {"email": request.email},
            {
                "$set": {
                    "biometric_enabled": True,
                    f"biometric_tokens.{request.device_id}": request.biometric_token,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"✅ Biometric enabled for: {request.email}")
        
        return {
            "success": True,
            "message": "Biometric authentication enabled"
        }
        
    except Exception as e:
        logger.error(f"❌ Biometric setup error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/biometric/login")
async def biometric_login(request: BiometricLoginRequest):
    """
    Login using biometric authentication
    """
    try:
        db = await get_db()
        
        # Find user and verify biometric token
        user = await db.users.find_one({
            "email": request.email,
            f"biometric_tokens.{request.device_id}": request.biometric_token
        })
        
        if not user:
            raise HTTPException(status_code=401, detail="Biometric authentication failed")
        
        # Update last login
        await db.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "last_login": datetime.utcnow()
                }
            }
        )
        
        # Generate tokens
        user_id = str(user["_id"])
        tokens = create_access_token(user_id, request.email, remember_me=True)
        
        logger.info(f"✅ Biometric login successful: {request.email}")
        
        return {
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user_id,
                "email": user["email"],
                "name": user["name"]
            },
            **tokens
        }
        
    except Exception as e:
        logger.error(f"❌ Biometric login error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/password-reset/init")
async def password_reset_init(request: PasswordResetInitRequest):
    """
    Step 1 of password reset: Send OTP to email
    """
    try:
        db = await get_db()
        
        # Check if user exists
        user = await db.users.find_one({"email": request.email})
        if not user:
            # Don't reveal if email exists or not for security
            return {
                "success": True,
                "message": "If this email is registered, you will receive a password reset code"
            }
        
        # Generate OTP
        otp = generate_otp()
        otp_expiry = datetime.utcnow() + timedelta(minutes=OTP_EXPIRATION_MINUTES)
        
        # Store OTP
        await db.password_resets.update_one(
            {"email": request.email},
            {
                "$set": {
                    "email": request.email,
                    "otp": otp,
                    "otp_expiry": otp_expiry,
                    "created_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        
        # Send OTP email
        email_subject = "Password Reset Code - EAiSER"
        email_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background: #f4f4f4; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 10px;">
                <h1 style="color: #f6c521;">Password Reset</h1>
                <p>You requested to reset your password.</p>
                <p>Your verification code is:</p>
                <div style="background: #f6c521; color: white; font-size: 32px; font-weight: bold; text-align: center; padding: 20px; border-radius: 8px; letter-spacing: 8px;">
                    {otp}
                </div>
                <p style="color: #666; margin-top: 20px;">This code will expire in {OTP_EXPIRATION_MINUTES} minutes.</p>
                <p style="color: #666;">If you didn't request this, please ignore this email.</p>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Password Reset
        
        You requested to reset your password.
        
        Your verification code is: {otp}
        
        This code will expire in {OTP_EXPIRATION_MINUTES} minutes.
        
        If you didn't request this, please ignore this email.
        """
        
        await send_email(
            to_email=request.email,
            subject=email_subject,
            html_content=email_body,
            text_content=text_content
        )
        
        logger.info(f"✅ Password reset OTP sent to {request.email}")
        
        return {
            "success": True,
            "message": "If this email is registered, you will receive a password reset code"
        }
        
    except Exception as e:
        logger.error(f"❌ Password reset init error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/password-reset/verify")
async def password_reset_verify(request: PasswordResetVerifyRequest):
    """
    Step 2 of password reset: Verify OTP and set new password
    """
    try:
        db = await get_db()
        
        # Get password reset request
        reset_request = await db.password_resets.find_one({"email": request.email})
        if not reset_request:
            raise HTTPException(status_code=404, detail="No password reset request found")
        
        # Verify OTP
        if reset_request["otp"] != request.otp:
            raise HTTPException(status_code=400, detail="Invalid verification code")
        
        # Check OTP expiry
        if datetime.utcnow() > reset_request["otp_expiry"]:
            raise HTTPException(status_code=400, detail="Verification code expired")
        
        # Update password
        await db.users.update_one(
            {"email": request.email},
            {
                "$set": {
                    "password_hash": hash_password(request.new_password),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Delete reset request
        await db.password_resets.delete_one({"email": request.email})
        
        logger.info(f"✅ Password reset successful for: {request.email}")
        
        return {
            "success": True,
            "message": "Password reset successful. You can now login with your new password."
        }
        
    except Exception as e:
        logger.error(f"❌ Password reset verify error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/refresh")
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token
    """
    try:
        # Verify refresh token
        payload = verify_token(request.refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Generate new tokens
        tokens = create_access_token(
            payload["user_id"],
            payload["email"],
            remember_me=True
        )
        
        return {
            "success": True,
            **tokens
        }
        
    except Exception as e:
        logger.error(f"❌ Token refresh error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current user information
    """
    try:
        db = await get_db()
        
        user = await db.users.find_one({"email": current_user["email"]})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "success": True,
            "user": {
                "id": str(user["_id"]),
                "email": user["email"],
                "name": user["name"],
                "created_at": user["created_at"].isoformat(),
                "biometric_enabled": user.get("biometric_enabled", False),
                "devices": user.get("devices", [])
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Get user info error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout (client should delete tokens)
    """
    logger.info(f"✅ User logged out: {current_user['email']}")
    
    return {
        "success": True,
        "message": "Logged out successfully"
    }
