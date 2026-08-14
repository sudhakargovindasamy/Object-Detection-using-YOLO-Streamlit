from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import io
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import glob
from pathlib import Path
import shutil
import uuid
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YOLO Object Detection API",
    description="API for object detection using YOLOv8",
    version="1.0.0"
)

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache = {}
DEFAULT_MODEL = "yolov8n.pt"

def get_model(model_name: str = DEFAULT_MODEL) -> YOLO:
    """Get or load YOLO model with caching"""
    if model_name not in model_cache:
        logger.info(f"Loading YOLO model: {model_name}")
        try:
            model_cache[model_name] = YOLO(model_name)
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    return model_cache[model_name]

@app.get("/")
async def root():
    return {"message": "YOLO Object Detection API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/models")
async def list_available_models():
    """List available YOLO models"""
    return {
        "models": [
            "yolov8n.pt",  # nano - fastest, smallest
            "yolov8s.pt",  # small
            "yolov8m.pt",  # medium
            "yolov8l.pt",  # large
            "yolov8x.pt",  # xlarge - most accurate
        ],
        "default": DEFAULT_MODEL
    }

@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    conf: float = Form(0.25),
    imgsz: int = Form(640)
):
    """
    Detect objects in an uploaded image.
    Returns annotated image and detection results.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # Load model
    try:
        yolo_model = get_model(model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    # Run inference
    try:
        results = yolo_model.predict(img_array, conf=conf, imgsz=imgsz, verbose=False)
        res = results[0]

        # Get annotated image
        plotted = res.plot()
        annotated = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        # Save annotated image to temp file
        temp_dir = tempfile.mkdtemp()
        annotated_path = os.path.join(temp_dir, f"annotated_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(annotated_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        # Extract detections
        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            for c, cf, box in zip(res.boxes.cls.cpu().numpy(),
                                   res.boxes.conf.cpu().numpy(),
                                   res.boxes.xyxy.cpu().numpy()):
                name = yolo_model.names[int(c)]
                detections.append({
                    "class": name,
                    "confidence": float(cf),
                    "bbox": [float(x) for x in box]  # x1, y1, x2, y2
                })

        return {
            "detections": detections,
            "annotated_image_url": f"/files/{os.path.basename(annotated_path)}",
            "image_shape": img_array.shape[:2],
            "model": model,
            "inference_time_ms": float(res.speed.get("inference", 0)) if hasattr(res, 'speed') else 0
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        # Cleanup temp dir after a delay (handled by cleanup endpoint or scheduled)
        pass

@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    conf: float = Form(0.25),
    imgsz: int = Form(640)
):
    """
    Detect objects in an uploaded video.
    Note: Video processing may take time. Consider using async job queue for production.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Check file size (limit to 50MB for Vercel/Render free tier)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video file too large (max 50MB)")

    # Save video to temp file
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}{Path(file.filename).suffix}")
    with open(video_path, "wb") as f:
        f.write(contents)

    try:
        yolo_model = get_model(model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    # Process video
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video
        output_path = os.path.join(temp_dir, f"output_{uuid.uuid4().hex}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        all_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = yolo_model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
            res = results[0]

            # Get annotated frame
            plotted = res.plot()
            annotated = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            # Extract detections
            if res.boxes is not None and len(res.boxes) > 0:
                frame_detections = []
                for c, cf, box in zip(res.boxes.cls.cpu().numpy(),
                                       res.boxes.conf.cpu().numpy(),
                                       res.boxes.xyxy.cpu().numpy()):
                    name = yolo_model.names[int(c)]
                    frame_detections.append({
                        "class": name,
                        "confidence": float(cf),
                        "bbox": [float(x) for x in box]
                    })
                all_detections.append({
                    "frame": frame_count,
                    "detections": frame_detections
                })

            frame_count += 1

            # Progress logging
            if frame_count % 50 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        out.release()

        return {
            "message": "Video processed successfully",
            "output_video_url": f"/files/{os.path.basename(output_path)}",
            "total_frames": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "detections": all_detections,
            "model": model
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    finally:
        # Cleanup will happen via cleanup endpoint or scheduled task
        pass

@app.get("/files/{filename}")
async def serve_file(filename: str):
    """Serve temporary files"""
    # Search in temp directories
    for temp_root in [tempfile.gettempdir(), "/tmp"]:
        for root, dirs, files in os.walk(temp_root):
            if filename in files:
                file_path = os.path.join(root, filename)
                return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/cleanup")
async def cleanup_temp_files():
    """Clean up old temporary files (call periodically)"""
    cleaned = 0
    for temp_root in [tempfile.gettempdir(), "/tmp"]:
        for item in os.listdir(temp_root):
            item_path = os.path.join(temp_root, item)
            try:
                # Only clean our temp dirs (those starting with tmp or containing our patterns)
                if item.startswith("tmp") or "annotated_" in item or "output_" in item or "input_" in item:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                    cleaned += 1
            except:
                pass
    return {"cleaned": cleaned}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)