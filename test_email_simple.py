
import asyncio
import os
from dotenv import load_dotenv
from app.services.email_service import send_email

# Load env variables
load_dotenv(override=True)

async def test_email():
    print("Testing email sending...")
    email_user = os.getenv("EMAIL_USER")
    api_key = os.getenv("SENDGRID_API_KEY")
    
    print(f"EMAIL_USER: {email_user}")
    print(f"SENDGRID_API_KEY: {api_key[:5]}..." if api_key else "SENDGRID_API_KEY: None")
    
    success = await send_email(
        to_email="chrishabh2002@gmail.com", # Using authority email from user request logs
        subject="Test Email from EAiSER Debugger",
        html_content="<h1>This is a test email</h1><p>If you see this, email sending works.</p>",
        text_content="This is a test email. If you see this, email sending works."
    )
    
    if success:
        print("✅ Email sent successfully via SendGrid!")
    else:
        print("❌ Email failed to send.")

if __name__ == "__main__":
    asyncio.run(test_email())
