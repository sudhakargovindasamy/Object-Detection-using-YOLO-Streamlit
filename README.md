# ��� Object Detection using YOLO + Next.js + FastAPI

This project demonstrates **real-time object detection** using the **YOLOv8 model** with a modern **Next.js frontend** deployed on **Vercel** and a **FastAPI backend** for ML inference.

---

## ������ Architecture

```
��─────────────────��     ��─────────────────��
│   Vercel        │     │   Render /      │
│   (Frontend)    │────��│   Railway /     │
│   Next.js 14    │     │   Fly.io        │
│                 │     │   FastAPI       │
��─────────────────��     └─────────────────��
                              │
                              ��
                        ��─────────────────��
                        │   Ultralytics   │
                        │   YOLOv8        │
                        │   (Auto-download)│
                        └─────────────────��
```

### Why this architecture?

- **Vercel** excels at hosting static/frontend applications (Next.js) but cannot run long-running Python processes
- **Streamlit** requires a persistent Python server, which Vercel's serverless functions cannot provide
- **FastAPI** on Render/Railway/Fly.io provides a proper Python backend for ML inference
- **YOLO models** are downloaded automatically by Ultralytics on first run (no need to bundle)

---

## ��� Project Structure

```
Object-Detection-using-YOLO-Streamlit/
├── frontend/                 # Next.js 14 frontend (Vercel)
│   ├── app/                  # App Router pages
│   ├── components/           # React components
│   ├── public/               # Static assets
│   ├── package.json
│   ├── next.config.js
│   ├── tsconfig.json
│   ├── vercel.json
│   └── .env.example
├── backend/                  # FastAPI backend (Render/Railway/Fly.io)
│   ├── main.py               # FastAPI application
│   ├── requirements.txt
│   └── Dockerfile
├── Object_Detection.ipynb    # Original notebook (reference)
├── app_streamlit.py          # Original Streamlit app (reference)
├── requirements.txt          # Python dependencies (reference)
├── .gitignore
├── .env.example
��── DEPLOYMENT.md
```

---

## ������ Technologies Used

### Frontend (Vercel)
- Next.js 14 (App Router)
- React 18
- TypeScript
- Tailwind-like CSS (custom)

### Backend (Render/Railway/Fly.io)
- FastAPI 0.109
- Uvicorn
- Ultralytics YOLOv8
- OpenCV
- PyTorch (via ultralytics)

---

## ��� Requirements

### Frontend
```bash
cd frontend
npm install
```

### Backend
```bash
cd backend
pip install -r requirements.txt
```

---

## ��� Local Development

### 1. Start Backend (Terminal 1)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Start Frontend (Terminal 2)
```bash
cd frontend
npm install
npm run dev
```

### 3. Open http://localhost:3000

The frontend will proxy API requests to `http://localhost:8000`.

---

## ��� Deployment

### Backend Deployment (Required First)

#### Option A: Render (Free Tier)
1. Connect GitHub repo to Render
2. Create new **Web Service**
3. Settings:
   - **Root Directory**: `backend`
   - **Runtime**: Docker (uses `backend/Dockerfile`)
   - **Instance Type**: Free
4. Deploy!
5. Copy the service URL (e.g., `https://yolo-backend.onrender.com`)

#### Option B: Railway
1. Connect GitHub repo to Railway
2. Add service from `backend/` directory
3. Uses `Dockerfile` automatically
4. Copy the service URL

#### Option C: Fly.io
```bash
cd backend
fly launch
fly deploy
```

### Frontend Deployment (Vercel)

1. Go to [Vercel](https://vercel.com)
2. Import the GitHub repository
3. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
4. Add Environment Variable:
   - `NEXT_PUBLIC_API_URL` = Your backend URL (e.g., `https://yolo-backend.onrender.com`)
5. Deploy!

---

## ��� Environment Variables

### Backend (Render/Railway/Fly.io)
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Auto | 8000 | Server port |

### Frontend (Vercel)
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | **Yes** | - | Backend API URL |

### .env.example (Frontend)
```bash
NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
```

---

## ��� Features

��� **Image Upload & Detection** - Drag & drop or click to upload images
��� **Real-time Bounding Boxes** - Visual annotations on detected objects
��� **Confidence Scores** - Visual confidence bars for each detection
��� **Multiple YOLO Models** - Nano, Small, Medium, Large, XLarge
��� **Adjustable Parameters** - Confidence threshold, image size
��� **Video Processing** - Upload and process videos (backend only)
��� **Responsive UI** - Works on desktop and mobile
��� **Error Handling** - User-friendly error messages
��� **Loading States** - Visual feedback during processing

---

## ��� API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/detect/image` | POST | Detect objects in image |
| `/detect/video` | POST | Detect objects in video |
| `/files/{filename}` | GET | Serve temporary files |
| `/cleanup` | POST | Clean up temp files |

---

## ������ Limitations

| Feature | Status | Notes |
|---------|--------|-------|
| Image Detection | �� Full Support | Works on Vercel + Backend |
| Video Detection | ������ Backend Only | Long processing time; not suitable for Vercel functions |
| Custom .pt Models | �� Supported | Upload via API (multipart/form-data) |
| Real-time Webcam | ��� Not Supported | Requires WebSocket/Streaming |
| GPU Acceleration | ������ Backend Only | Requires GPU-enabled backend (not free tier) |

---

## ��� Testing

### Test Image Detection
```bash
curl -X POST http://localhost:8000/detect/image \
  -F "file=@test.jpg" \
  -F "model=yolov8n.pt" \
  -F "conf=0.25"
```

### Test Video Detection
```bash
curl -X POST http://localhost:8000/detect/video \
  -F "file=@test.mp4" \
  -F "model=yolov8n.pt"
```

---

## ��� License

MIT License - feel free to use for personal or commercial projects.

---

## ���‍���� Author

**Sudhakar Govindasamy**

���� GitHub: [sudhakargovindasamy](https://github.com/sudhakargovindasamy)  
���� LinkedIn: [Sudhakar](https://www.linkedin.com/in/sudhakargovindasamy)