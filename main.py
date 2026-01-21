#cv intern task: object detection+FPS comparision
# model: YOLOv5s
import cv2
import torch
import time

# Load the pre-trained YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # detection confidence threshold

# Open the traffic video
video_capture = cv2.VideoCapture('task.mp4')

# Setup for FPS calculation
start_time = time.time()
frame_counter = 0

# Change this for resolution tests: 640 or 1280
frame_size = 640

print("Starting object detection... Press 'q' to quit.")

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        print("Video finished or cannot read the frame.")
        break

    # Resize the frame to the chosen resolution
    frame = cv2.resize(frame, (frame_size, frame_size))

    # Run object detection
    detection_results = model(frame)

    # Get annotated frame and make it writable
    annotated_frame = detection_results.render()[0].copy()

    # Update frame count and calculate FPS
    frame_counter += 1
    elapsed_time = time.time() - start_time
    fps = frame_counter / elapsed_time if elapsed_time > 0 else 0

    # Display FPS on the video
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # Show the annotated video
    cv2.imshow("Object Detection", annotated_frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Detection stopped by user.")
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
print("Video processing finished.")