import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YOLO Object Detection",
  description: "Real-time object detection using YOLOv8",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}