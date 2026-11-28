
import cv2
import time
import statistics
from ultralytics import YOLO

PT_MODEL = "yolo11n.pt"
ONNX_MODEL = "yolo11n.onnx"
VIDEO_PATH = "sample.mp4"

IMAGE_SIZE = 640
WARMUP = 10
ITERATIONS = 100

# Read one identical frame for both models
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read sample.mp4")

print("Loading PyTorch model...")
pt_model = YOLO(PT_MODEL)

print("Loading ONNX model...")
onnx_model = YOLO(ONNX_MODEL, task="detect")

def run_benchmark(model, name):
    print(f"\nWarming up {name}...")

    for _ in range(WARMUP):
        model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
        )

    times = []

    print(f"Testing {name}...")

    for _ in range(ITERATIONS):
        start = time.perf_counter()

        model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
        )

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    average = statistics.mean(times)

    return {
        "average": average,
        "median": statistics.median(times),
        "fps": 1000 / average
    }

pt = run_benchmark(pt_model, "PyTorch")
onnx = run_benchmark(onnx_model, "ONNX Runtime")

print("\n========== PYTORCH vs ONNX ==========")

print("\nPYTORCH (.pt)")
print(f"Average latency: {pt['average']:.2f} ms")
print(f"Median latency:  {pt['median']:.2f} ms")
print(f"FPS:             {pt['fps']:.2f}")

print("\nONNX RUNTIME (.onnx)")
print(f"Average latency: {onnx['average']:.2f} ms")
print(f"Median latency:  {onnx['median']:.2f} ms")
print(f"FPS:             {onnx['fps']:.2f}")

ratio = pt["average"] / onnx["average"]
reduction = (pt["average"] - onnx["average"]) / pt["average"] * 100

print("\n========== COMPARISON ==========")
print(f"Latency ratio: {ratio:.2f}x")
print(f"Latency reduction: {reduction:.2f}%")

if ratio > 1:
    print("ONNX was faster in this test.")
elif ratio < 1:
    print("PyTorch was faster in this test.")
else:
    print("Both had equal average latency.")

with open("onnx_comparison_results.txt", "w") as f:
    f.write("VISIONAI PYTORCH VS ONNX BENCHMARK\n")
    f.write(f"Input size: {IMAGE_SIZE}\n")
    f.write(f"Iterations: {ITERATIONS}\n")
    f.write(f"PyTorch latency: {pt['average']:.2f} ms\n")
    f.write(f"PyTorch FPS: {pt['fps']:.2f}\n")
    f.write(f"ONNX latency: {onnx['average']:.2f} ms\n")
    f.write(f"ONNX FPS: {onnx['fps']:.2f}\n")
    f.write(f"Latency reduction: {reduction:.2f}%\n")

print("\nResults saved to onnx_comparison_results.txt")
