import cv2
import time
import platform
import statistics
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "sample.mp4"

WARMUP_FRAMES = 10
MAX_FRAMES = 300

# ---------------- LOAD MODEL ----------------

print("Loading YOLO11n...")
model = YOLO(MODEL_PATH)

print("Model loaded successfully.")

# ---------------- OPEN VIDEO ----------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_video_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps_original = cap.get(cv2.CAP_PROP_FPS)

print("\nVideo information")
print("-----------------")
print(f"Resolution: {width} x {height}")
print(f"Original video FPS: {fps_original:.2f}")
print(f"Total video frames: {total_video_frames}")

# ---------------- WARMUP ----------------

print("\nRunning warmup...")

for _ in range(WARMUP_FRAMES):
    ret, frame = cap.read()

    if not ret:
        break

    model.predict(frame, verbose=False)

# ---------------- BENCHMARK ----------------

inference_times = []
processed_frames = 0

print("\nStarting benchmark...")

start_total = time.perf_counter()

while processed_frames < MAX_FRAMES:
    ret, frame = cap.read()

    if not ret:
        break

    start = time.perf_counter()

    model.predict(frame, verbose=False)

    end = time.perf_counter()

    inference_times.append((end - start) * 1000)
    processed_frames += 1

end_total = time.perf_counter()

cap.release()

# ---------------- RESULTS ----------------

if not inference_times:
    raise RuntimeError("No frames were processed.")

total_time = end_total - start_total
avg_ms = statistics.mean(inference_times)
median_ms = statistics.median(inference_times)
min_ms = min(inference_times)
max_ms = max(inference_times)

inference_fps = 1000 / avg_ms
end_to_end_fps = processed_frames / total_time

print("\n========== BENCHMARK RESULTS ==========")
print(f"Model: YOLO11n")
print(f"Device: CPU (default)")
print(f"Processor: {platform.processor()}")
print(f"Operating system: {platform.system()}")
print(f"Video resolution: {width} x {height}")
print(f"Frames processed: {processed_frames}")
print("---------------------------------------")
print(f"Average inference time: {avg_ms:.2f} ms")
print(f"Median inference time: {median_ms:.2f} ms")
print(f"Minimum inference time: {min_ms:.2f} ms")
print(f"Maximum inference time: {max_ms:.2f} ms")
print(f"Inference FPS: {inference_fps:.2f}")
print(f"End-to-end FPS: {end_to_end_fps:.2f}")
print(f"Total benchmark time: {total_time:.2f} seconds")
print("=======================================")

# Save results
with open("benchmark_results.txt", "w") as file:
    file.write("VISIONAI YOLO11n BENCHMARK\n")
    file.write(f"Model: YOLO11n\n")
    file.write(f"Device: CPU\n")
    file.write(f"Processor: {platform.processor()}\n")
    file.write(f"Resolution: {width}x{height}\n")
    file.write(f"Frames processed: {processed_frames}\n")
    file.write(f"Average inference: {avg_ms:.2f} ms/frame\n")
    file.write(f"Inference FPS: {inference_fps:.2f}\n")
    file.write(f"End-to-end FPS: {end_to_end_fps:.2f}\n")

print("\nResults saved to benchmark_results.txt")