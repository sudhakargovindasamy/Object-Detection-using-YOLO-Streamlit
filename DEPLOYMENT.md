# Deployment Guide: Object Detection YOLO + Next.js + FastAPI

This document describes how to deploy the Object Detection app to production.

## Architecture Overview

```
Frontend (Vercel)          Backend (Render/Railway/Fly.io)
Next.js 14                 FastAPI + YOLOv8
     │                           │
     └────────── API ────────────��
```

**Vercel cannot run Streamlit or long-running Python processes.** This architecture separates the frontend (Vercel) from the ML inference backend (Render/Railway/Fly.io).

---

## Prerequisites

1. GitHub account: https://github.com/
2. Vercel account: https://vercel.com/
3. Backend hosting account (choose one):
   - Render: https://render.com/ (recommended - free tier)
   - Railway: https://railway.app/
   - Fly.io: https://fly.io/

---

## Step 1: Deploy Backend (Required First)

### Option A: Render (Recommended - Free Tier)

1. Go to https://dashboard.render.com/
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `yolo-object-detection-api`
   - **Root Directory**: `backend`
   - **Runtime**: `Docker`
   - **Dockerfile Path**: `backend/Dockerfile`
   - **Instance Type**: `Free`
5. Click "Create Web Service"
6. Wait for deployment (5-10 minutes for first build - downloads PyTorch/YOLO)
7. Copy the service URL (e.g., `https://yolo-object-detection-api.onrender.com`)

### Option B: Railway

1. Go to https://railway.app/
2. Click "New Project" → "Deploy from GitHub repo"
3. Select repository
4. Click "Add Service" → "Dockerfile"
5. Set **Dockerfile Path**: `backend/Dockerfile`
6. Deploy
7. Copy the service URL

### Option C: Fly.io

```bash
cd backend
fly launch --name yolo-api
fly deploy
```

---

## Step 2: Deploy Frontend to Vercel

1. Go to https://vercel.com/
2. Click "Add New..." → "Project"
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
   - **Install Command**: `npm install`
5. Add Environment Variable:
   - **Name**: `NEXT_PUBLIC_API_URL`
   - **Value**: Your backend URL from Step 1 (e.g., `https://yolo-object-detection-api.onrender.com`)
6. Click "Deploy"
7. Wait for build to complete
8. Open your Vercel URL

---

## Step 3: Test the Deployed Application

1. Open your Vercel URL
2. Select "Image" mode
3. Upload a test image
4. Click "Detect Objects"
5. Verify annotated image and detections appear

---

## Environment Variables Reference

### Frontend (Vercel)
| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | **Yes** | Backend API URL (e.g., `https://your-api.onrender.com`) |

### Backend (Render/Railway/Fly.io)
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Auto | 8000 | Server port (set automatically by platform) |

---

## Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Opens http://localhost:3000
```

---

## Testing API Endpoints

### Health Check
```bash
curl https://your-backend.onrender.com/health
```

### List Models
```bash
curl https://your-backend.onrender.com/models
```

### Image Detection
```bash
curl -X POST https://your-backend.onrender.com/detect/image \
  -F "file=@test.jpg" \
  -F "model=yolov8n.pt" \
  -F "conf=0.25" \
  -F "imgsz=640"
```

### Video Detection
```bash
curl -X POST https://your-backend.onrender.com/detect/video \
  -F "file=@test.mp4" \
  -F "model=yolov8n.pt"
```

---

## Troubleshooting

### Backend takes too long to start
- First deploy downloads PyTorch and YOLO (~2GB). Subsequent deploys are faster.
- Render free tier spins down after 15 min inactivity. First request after spin-down takes 30-60s.

### CORS errors
- Backend allows all origins (`*`) by default. For production, update `allow_origins` in `backend/main.py` to your Vercel domain.

### Video processing times out
- Video processing is synchronous and may exceed platform timeouts (30-60s).
- For production video processing, implement async job queue (Celery + Redis).

### Model download fails
- Ensure backend has internet access to download from GitHub/Ultralytics.
- Check firewall/proxy settings.

---

## Costs

| Platform | Free Tier | Paid Tier |
|----------|-----------|-----------|
| Vercel | 100GB bandwidth, unlimited personal projects | $20/mo Pro |
| Render | 750 hrs/mo, spins down after 15min idle | $7/mo |
| Railway | $5 credit/mo (then pay-as-you-go) | Pay-as-you-go |
| Fly.io | 3 shared-cpu VMs, 160GB bandwidth | Pay-as-you-go |

**Estimated cost: $0/month** on free tiers for low-traffic usage.

---

## Security Notes

- No API keys or secrets required for basic operation
- YOLO models downloaded from official Ultralytics releases
- CORS configured for development; restrict in production
- No authentication implemented (add if needed)
- Temporary files cleaned up periodically