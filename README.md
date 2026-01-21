# Computer Vision Intern Task – Object Detection Pipeline

This repository contains a simple computer vision pipeline built for an internship task. The pipeline detects objects in a short driving video using a **pretrained YOLOv5 model**.

---

## Features

- **Object Detection**: Detects vehicles (cars, trucks, buses, etc.) in a driving video.
- **FPS Measurement**: Pipeline calculates approximate frames per second (FPS) at different resolutions.
- **Easy to Run**: Works on Linux systems with Python, OpenCV, and PyTorch installed.
- **Pretrained Model**: Uses YOLOv5s pretrained weights to save time and focus on pipeline and performance evaluation.

---

## Installation

1. Clone the repository (or download files manually if using GitHub “Add file”):

git clone https://github.com/nithya881/cv_interntask.git

2. Install required Python packages:

pip install opencv-python torch torchvision numpy

3. Ensure you have a short vedio.

---

## Usage

Open main.py and set resolution
frame_size = 640 #or 1280 for higher resolution

Run the code
python main.py
- A window will show the video with detected objects and FPS displayed on each frame.  
- Press `q` to exit early.

---

## Notes on Accuracy

False positives may occur: e.g., auto rickshaws may be detected as cars or trucks because they are not part of the YOLOv5 pretrained dataset.
Most cars, trucks, and buses are correctly detected.
FPS vs accuracy was measured at multiple resolutions to evaluate edge performance.

---

## Optimization

The code runs on CPU by default, but GPU acceleration is supported if PyTorch is installed with CUDA.
For better FPS, lower resolutions can be used (e.g., 640 × 640).

---

## Folder Structure


CV_task/
├── main.py            # Main Python script 
├── input_video.mp4    # Sample short video
├── README.md          # This file 
└── .gitignore         # Ignore unnecessary files like weights or large videos























