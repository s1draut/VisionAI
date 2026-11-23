# VisionAI – Universal Video Intelligence System

An AI-powered computer vision application for image and video object detection, built using **YOLO11n, OpenCV, Python and Streamlit**.

**Live Demo:** https://vdetect.streamlit.app  
**GitHub Repository:** https://github.com/s1draut/VisionAI

---

## Overview

VisionAI is a computer vision application that analyses images and videos to identify and localize objects using a pretrained YOLO11n object detection model.

The application provides visual detection results, object-count analytics, frame-level detection records and downloadable reports through an interactive web interface.

The project demonstrates the practical integration of deep learning-based object detection with video processing and data analytics.

## Features

### 1. Image Detection
- Upload JPG, JPEG and PNG images.
- Detect multiple objects using YOLO11n.
- Display bounding boxes and confidence scores.
- Generate object-count analytics.
- Export detection results as CSV.

### 2. Video Detection
- Upload MP4, AVI, MOV and MKV videos.
- Perform frame-sampled object detection.
- Configure frame sampling intervals.
- Generate annotated videos with H.264 encoding.
- Display frame-level detection records and timestamps.
- Download annotated videos and CSV reports.

### 3. Analytics
- Total object detections.
- Unique object classes.
- Object-wise detection counts.
- Confidence scores.
- Frame and timestamp information.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO11n | Object detection |
| Ultralytics | Model inference framework |
| OpenCV | Image and video processing |
| Streamlit | Web application interface |
| FFmpeg | H.264 video encoding |
| Pandas / CSV | Data handling and reporting |
| Pillow | Image processing |

## System Architecture

```text
        User Input
            |
     Image / Video Upload
            |
       Streamlit UI
            |
       YOLO11n Model
            |
    Object Detection
            |
    +-------+-------+
    |               |
 Bounding Boxes   Detection Data
    |               |
 Annotated Media  Analytics
    |               |
    +-------+-------+
            |
       CSV Reports
            |
       User Output
```

## Model Information

- Model: YOLO11n (Nano)
- Framework: Ultralytics
- Model weights: Pretrained `yolo11n.pt`
- Dataset classes: COCO (80 classes)
- Inference mode: CPU
- Model training: Not performed as part of this project.

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/s1draut/VisionAI.git
cd VisionAI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

Open the application at:

```text
http://localhost:8501
```

## Limitations

- Detection is limited to the classes supported by the pretrained COCO model.
- Detection performance depends on image quality, lighting, object size and confidence threshold.
- Video analysis uses configurable frame sampling rather than analysing every frame by default.
- Processing speed depends on available hardware.
- Audio is not preserved in the generated annotated video.

## Future Enhancements

- Object tracking across video frames.
- Real-time webcam detection.
- Performance benchmarking using FPS and inference latency.
- Advanced detection analytics and visualizations.
- Custom dataset training and model fine-tuning.
- Improved video processing and scalability.

## Project Links

- **Live Application:** https://vdetect.streamlit.app
- **Source Code:** https://github.com/s1draut/VisionAI

---

**VisionAI**  
Computer Vision | Deep Learning | Object Detection | Video Analytics
