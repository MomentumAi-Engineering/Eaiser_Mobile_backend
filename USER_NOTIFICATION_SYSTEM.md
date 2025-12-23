# User Notification System for Admin Actions

## Overview
When admin approves or declines a report, the user who submitted it will receive an email notification.

## Email Templates

### 1. Report Approved Email
**Subject:** ✅ Your Report Has Been Approved - EAiSER

**Body:**
```
Dear User,

Good news! Your report has been reviewed and APPROVED by our admin team.

Report Details:
- Report ID: {issue_id}
- Status: Approved
- Action Taken: Your report has been forwarded to the relevant authorities
- Next Steps: The authorities will investigate and take necessary action

Thank you for helping make our community safer!

Best regards,
EAiSER Admin Team
```

### 2. Report Declined Email
**Subject:** ❌ Your Report Status Update - EAiSER

**Body:**
```
Dear User,

Your report has been reviewed by our admin team.

Report Details:
- Report ID: {issue_id}
- Status: Declined
- Reason: {decline_reason}

If you believe this was a mistake, please submit a new report with more details.

Thank you for your understanding.

Best regards,
EAiSER Admin Team
```

## Implementation

### Backend Changes:
1. **Email Service:** Uses existing `send_email` function from `services/email_service.py`
2. **Notification Trigger:** 
   - When admin clicks "Approve" → Send approval email
   - When admin clicks "Decline" → Send decline email with reason
3. **User Email:** Retrieved from the report's `user_email` or `reporter_email` field

### Frontend Changes:
1. **Decline Button:** Prompts admin for decline reason
2. **Approve Button:** Confirms approval action
3. **Success Message:** Shows confirmation that user has been notified

## Testing
1. Submit a test report from mobile app
2. Go to admin dashboard
3. Click "Decline" → Enter reason → User receives decline email
4. Click "Approve" → User receives approval email

## Status
- ✅ Email templates defined
- 🔄 Backend implementation in progress
- ⏳ Frontend integration pending
