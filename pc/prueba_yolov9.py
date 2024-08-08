import torch
import numpy as np
import cv2
from ultralytics import YOLO  # This is how you import the YOLO model from the ultralytics package
import time

# Load the model
model_path = 'yolov9c.pt'  # Pre-trained model; replace with your model path if needed
#model_path = 'yolov8m.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path)
#model.to(device).eval() no estoy seguro de necesitar aca

# Setup UDP stream (replace 'IP' and 'PORT' with your actual values)
udp_url = 'udp://@IP:PORT'
cap = cv2.VideoCapture(0)

# Initialize timing statistics
times = []
print("Iniciando")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and then to a torch tensor
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).to(device).float().div(255.0)
        image = image.permute(2, 0, 1).unsqueeze(0)  # Reorder dimensions to CxHxW and add batch dimension

        # Inference
        start_time = time.time()
        with torch.no_grad():
            pred = model(image)
        elapsed = time.time() - start_time
        times.append(elapsed)

        # Print current frame's inference time
        print(f"Inference time for current frame: {elapsed:.3f} seconds")

        # Display the processed frame (optional)
        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
            break

        # Update and print statistics
        if len(times) % 10 == 0:  # Update statistics every 10 frames
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            print(f"Updated every 10 frames - Average: {avg_time:.3f}, Min: {min_time:.3f}, Max: {max_time:.3f}")

finally:
    cap.release()
    cv2.destroyAllWindows()

# Final statistics
avg_time = np.mean(times)
min_time = np.min(times)
max_time = np.max(times)

print(f"Final Average Inference Time: {avg_time:.3f} seconds")
print(f"Final Minimum Inference Time: {min_time:.3f} seconds")
print(f"Final Maximum Inference Time: {max_time:.3f} seconds")
