
# 🚀 How to Deploy to Render

I have prepared your repository for deployment.

1.  **Push Changes:**
    First, verify you have pushed the latest files (including `render.yaml` and `requirements.txt`):
    ```bash
    git add .
    git commit -m "Prepare for Render Deployment"
    git push
    ```

2.  **Go to Render:**
    *   Log in to [dashboard.render.com](https://dashboard.render.com).
    *   Click **New +** -> **Web Service**.
    *   Connect your GitHub repository: `MomentumAi-Engineering/Eaiser_Mobile_backend`.

3.  **Configure:**
    *   Render will automatically detect `render.yaml` and fill in the settings!
    *   **Runtime:** Python 3
    *   **Build Command:** `pip install -r requirements.txt`
    *   **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

4.  **🛑 IMPORTANT - Environmental Variables:**
    You MUST add your keys in the "Environment" tab on Render:
    *   `MONGO_URI`: (Your MongoDB Connection String)
    *   `OPENAI_API_KEY`: (Your AI Key)
    *   `GEMINI_API_KEY`: (Your Gemini Key)
    *   `SENDGRID_API_KEY`: (SG....)
    *   `EMAIL_USER`: (e.g. no-reply@eaiser.ai)
    *   ...and any others from your `.env` file.

5.  **Click Deploy!** 🚀
