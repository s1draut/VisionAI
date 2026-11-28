
from ultralytics import YOLO
import time
import os

print("Loading YOLO11n...")
model = YOLO("yolo11n.pt")

print("\nExporting model to ONNX...")

start = time.perf_counter()

onnx_path = model.export(
    format="onnx",
    imgsz=640,
    simplify=False,
    device="cpu"
)

export_time = time.perf_counter() - start

print("\n========== EXPORT RESULTS ==========")
print("Export successful!")
print("ONNX file:", onnx_path)
print(f"Export duration: {export_time:.2f} seconds")

if os.path.exists(onnx_path):
    size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX file size: {size:.2f} MB")
else:
    print("Warning: ONNX file not found.")

print("====================================")