
import streamlit as st
import cv2
import tempfile
import os
import csv
import io
import subprocess
import imageio_ffmpeg
from collections import Counter
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="VisionAI", page_icon="👁️", layout="wide")

st.title("VisionAI")
st.caption("Universal Video Intelligence System | AI-powered object detection")


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()

mode = st.sidebar.radio(
    "Select Mode",
    ["Image Detection", "Video Detection"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.35, 0.05
)


if mode == "Image Detection":

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner("Detecting objects..."):
            results = model.predict(
                image,
                conf=confidence,
                device="cpu",
                verbose=False
            )

        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detection Results")
            st.image(annotated, use_container_width=True)

        boxes = results[0].boxes
        names = model.names
        counts = Counter()
        rows = []

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                label = names[class_id]
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                counts[label] += 1
                rows.append([
                    label, round(score, 4),
                    x1, y1, x2, y2
                ])

        a, b = st.columns(2)
        a.metric("Objects detected", len(rows))
        b.metric("Unique classes", len(counts))

        st.subheader("Object Analytics")

        if counts:
            st.bar_chart(dict(counts))
            st.dataframe(
                [
                    {"Object": k, "Count": v}
                    for k, v in counts.items()
                ],
                use_container_width=True
            )

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Object", "Confidence",
            "X1", "Y1", "X2", "Y2"
        ])
        writer.writerows(rows)

        st.download_button(
            "Download CSV Report",
            output.getvalue(),
            file_name="visionai_image_report.csv",
            mime="text/csv"
        )


else:

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    frame_step = st.selectbox(
        "Analyse every Nth frame",
        [1, 2, 5, 10],
        index=2
    )

    if uploaded and st.button("Analyse Video"):

        input_path = None
        output_path = None
        process = None

        try:
            suffix = os.path.splitext(uploaded.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp:
                temp.write(uploaded.read())
                input_path = temp.name

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                raise RuntimeError(
                    "Could not open the uploaded video."
                )

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if fps <= 0:
                fps = 25.0

            if width <= 0 or height <= 0:
                raise RuntimeError(
                    "The video dimensions could not be read."
                )

            output_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            )
            output_path = output_file.name
            output_file.close()

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

            command = [
                ffmpeg,
                "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-an",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ]

            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            all_rows = []
            counts = Counter()
            frame_number = 0
            analysed_frames = 0

            progress = st.progress(0)
            status = st.empty()

            while True:
                success, frame = cap.read()

                if not success:
                    break

                if frame_number % frame_step == 0:

                    results = model.predict(
                        frame,
                        conf=confidence,
                        device="cpu",
                        verbose=False
                    )

                    annotated = results[0].plot()

                    boxes = results[0].boxes

                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            label = model.names[class_id]
                            score = float(box.conf[0])

                            x1, y1, x2, y2 = map(
                                int, box.xyxy[0].tolist()
                            )

                            counts[label] += 1

                            all_rows.append([
                                frame_number,
                                round(frame_number / fps, 2),
                                label,
                                round(score, 4),
                                x1, y1, x2, y2
                            ])

                    process.stdin.write(
                        annotated.tobytes()
                    )

                    analysed_frames += 1

                else:
                    process.stdin.write(frame.tobytes())

                frame_number += 1

                if total_frames > 0:
                    progress.progress(
                        min(frame_number / total_frames, 1.0)
                    )

                status.text(
                    f"Processed {frame_number} / "
                    f"{total_frames} frames"
                )

            cap.release()
            process.stdin.close()
            return_code = process.wait()
            process = None

            progress.empty()
            status.empty()

            if return_code != 0:
                raise RuntimeError(
                    "FFmpeg could not encode the output video."
                )

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError(
                    "The output video was not created."
                )

            st.success("Video analysis completed!")

            st.subheader("Annotated Video")
            st.video(output_path, format="video/mp4")

            a, b, c = st.columns(3)
            a.metric("Frames processed", analysed_frames)
            b.metric("Total detections", len(all_rows))
            c.metric("Unique classes", len(counts))

            st.subheader("Object Analytics")

            if counts:
                st.bar_chart(dict(counts))

            st.subheader("Detection Records")

            st.dataframe(
                [
                    {
                        "Frame": r[0],
                        "Timestamp": r[1],
                        "Object": r[2],
                        "Confidence": r[3],
                        "X1": r[4],
                        "Y1": r[5],
                        "X2": r[6],
                        "Y2": r[7]
                    }
                    for r in all_rows
                ],
                use_container_width=True
            )

            output = io.StringIO()
            csv_writer = csv.writer(output)

            csv_writer.writerow([
                "Frame", "Timestamp", "Object",
                "Confidence", "X1", "Y1", "X2", "Y2"
            ])

            csv_writer.writerows(all_rows)

            st.download_button(
                "Download Video CSV Report",
                output.getvalue(),
                file_name="visionai_video_report.csv",
                mime="text/csv"
            )

            with open(output_path, "rb") as video_file:
                st.download_button(
                    "Download Annotated Video",
                    video_file,
                    file_name="visionai_annotated_video.mp4",
                    mime="video/mp4"
                )

        except Exception as error:
            st.error(f"Video processing failed: {error}")

        finally:
            if process is not None:
                try:
                    process.kill()
                    process.wait()
                except Exception:
                    pass

            if input_path and os.path.exists(input_path):
                os.remove(input_path)
