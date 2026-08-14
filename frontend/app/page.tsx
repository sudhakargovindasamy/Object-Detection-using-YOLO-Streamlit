"use client";

import { useState, useRef, useCallback } from "react";

interface Detection {
  class: string;
  confidence: number;
  bbox: number[];
}

interface ImageDetectionResult {
  detections: Detection[];
  annotated_image_url: string;
  image_shape: [number, number];
  model: string;
  inference_time_ms: number;
}

interface VideoDetectionResult {
  message: string;
  output_video_url: string;
  total_frames: number;
  fps: number;
  width: number;
  height: number;
  detections: { frame: number; detections: Detection[] }[];
  model: string;
}

type DetectionResult = ImageDetectionResult | VideoDetectionResult;

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function isImageResult(result: DetectionResult): result is ImageDetectionResult {
  return "annotated_image_url" in result;
}

export default function Home() {
  const [mode, setMode] = useState<"image" | "video">("image");
  const [model, setModel] = useState("yolov8n.pt");
  const [conf, setConf] = useState(0.25);
  const [imgsz, setImgsz] = useState(640);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((file: File) => {
    setError(null);
    setResult(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.add("dragover");
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.currentTarget.classList.remove("dragover");
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const detectImage = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", model);
    formData.append("conf", conf.toString());
    formData.append("imgsz", imgsz.toString());

    try {
      const response = await fetch(`${API_URL}/detect/image`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Detection failed");
      }

      const data: ImageDetectionResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  const detectVideo = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setResult(null);
    setProgress({ current: 0, total: 100 });

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", model);
    formData.append("conf", conf.toString());
    formData.append("imgsz", imgsz.toString());

    try {
      const response = await fetch(`${API_URL}/detect/video`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Video processing failed");
      }

      const data: VideoDetectionResult = await response.json();
      setResult(data);
      setProgress(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      setProgress(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDetect = () => {
    if (mode === "image") {
      detectImage();
    } else {
      detectVideo();
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setProgress(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const isImage = mode === "image";
  const acceptedTypes = isImage ? "image/*" : "video/*";

  return (
    <div className="container">
      <h1>���� YOLO Object Detection</h1>

      <div className="card">
        <h2>Settings</h2>
        <div className="controls">
          <div className="control-group">
            <label>Mode</label>
            <select value={mode} onChange={(e) => setMode(e.target.value as "image" | "video")}>
              <option value="image">������� Image</option>
              <option value="video">���� Video</option>
            </select>
          </div>
          <div className="control-group">
            <label>Model</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="yolov8n.pt">YOLOv8 Nano (fastest)</option>
              <option value="yolov8s.pt">YOLOv8 Small</option>
              <option value="yolov8m.pt">YOLOv8 Medium</option>
              <option value="yolov8l.pt">YOLOv8 Large</option>
              <option value="yolov8x.pt">YOLOv8 XLarge (most accurate)</option>
            </select>
          </div>
          <div className="control-group">
            <label>Confidence Threshold: {conf.toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={conf}
              onChange={(e) => setConf(parseFloat(e.target.value))}
            />
          </div>
          <div className="control-group">
            <label>Image Size: {imgsz}px</label>
            <select value={imgsz} onChange={(e) => setImgsz(parseInt(e.target.value))}>
              <option value={320}>320</option>
              <option value={416}>416</option>
              <option value={640}>640</option>
              <option value={1280}>1280</option>
            </select>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>{isImage ? "Upload Image" : "Upload Video"}</h2>

        <div
          className="upload-area"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={triggerFileInput}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={acceptedTypes}
            onChange={handleInputChange}
            style={{ display: "none" }}
          />
          <p>Drag & drop or click to upload</p>
          <small>
            {isImage ? "JPG, PNG (max 10MB)" : "MP4, MOV, AVI (max 50MB)"}
          </small>
        </div>

        {previewUrl && (
          <div style={{ marginTop: "1rem" }}>
            {isImage ? (
              <img src={previewUrl} alt="Preview" style={{ maxWidth: "100%", maxHeight: "300px", borderRadius: "8px" }} />
            ) : (
              <video src={previewUrl} controls style={{ maxWidth: "100%", maxHeight: "300px", borderRadius: "8px" }} />
            )}
          </div>
        )}

        {selectedFile && (
          <div style={{ marginTop: "1rem" }}>
            <button
              className="btn btn-primary"
              onClick={handleDetect}
              disabled={isLoading}
            >
              {isLoading ? "��� Processing..." : isImage ? "���� Detect Objects" : "���� Process Video"}
            </button>
            <button
              className="btn btn-secondary"
              onClick={reset}
              style={{ marginLeft: "1rem" }}
              disabled={isLoading}
            >
              Clear
            </button>
          </div>
        )}

        {progress && (
          <div className="loading" style={{ marginTop: "1rem" }}>
            <div className="spinner" />
            <p>Processing video... Frame {progress.current} / {progress.total}</p>
          </div>
        )}

        {error && <div className="error">{error}</div>}
      </div>

      {result && (
        <div className="card">
          <h2>Results</h2>
          {isImage && isImageResult(result) && (
            <div className="results-grid">
              <div className="result-card">
                <h3>Annotated Image</h3>
                <img
                  src={`${API_URL}${result.annotated_image_url}`}
                  alt="Annotated"
                  className="result-image"
                />
                <p style={{ marginTop: "1rem", fontSize: "0.9rem", color: "#666" }}>
                  Model: {result.model} | Inference: {result.inference_time_ms.toFixed(1)}ms
                </p>
              </div>
              <div className="result-card">
                <h3>Detections ({result.detections.length})</h3>
                {result.detections.length > 0 ? (
                  <table className="detection-table">
                    <thead>
                      <tr>
                        <th>Class</th>
                        <th>Confidence</th>
                        <th>Bounding Box</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.detections.map((det, i) => (
                        <tr key={i}>
                          <td>
                            <span style={{ textTransform: "capitalize" }}>{det.class}</span>
                          </td>
                          <td>
                            <div style={{ width: "100px" }}>
                              <div
                                className="confidence-bar"
                                style={{ width: `${det.confidence * 100}%` }}
                              />
                            </div>
                            <span style={{ fontSize: "0.85rem" }}>{(det.confidence * 100).toFixed(1)}%</span>
                          </td>
                          <td>
                            <code style={{ fontSize: "0.75rem" }}>
                              [{det.bbox.map((v) => v.toFixed(1)).join(", ")}]
                            </code>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p style={{ color: "#888", textAlign: "center", padding: "2rem" }}>
                    No objects detected
                  </p>
                )}
              </div>
            </div>
          )}

          {result && !isImage && "output_video_url" in result && (
            <div className="results-grid">
              <div className="result-card">
                <h3>Processed Video</h3>
                <video
                  src={`${API_URL}${result.output_video_url}`}
                  controls
                  style={{ width: "100%" }}
                />
                <p style={{ marginTop: "1rem", fontSize: "0.9rem", color: "#666" }}>
                  {result.total_frames} frames @ {result.fps.toFixed(1)}fps | {result.width}x{result.height}
                </p>
              </div>
              <div className="result-card">
                <h3>Summary</h3>
                <p>Total frames processed: {result.total_frames}</p>
                <p>Total detections: {result.detections.reduce((sum, f) => sum + f.detections.length, 0)}</p>
                <p>Unique classes: {new Set(result.detections.flatMap((f) => f.detections.map((d) => d.class))).size}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
