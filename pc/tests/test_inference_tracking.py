import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
import torch
from torchvision.ops import nms
from collections import deque
import time

# Load the YOLO model and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not torch.cuda.is_available():
    print("Cuda not available :(")

model = YOLO(r'./pc/best_exp5.pt')  # Load your model
names = model.model.names

# Initialize video input and output
#input_video = r'C:\Users\josec\Documents\Github\fruit-ripeness-classificator\videos\naranja_videos3\video_20240914_222246.h264'  # Replace with your test video path
#cap = cv2.VideoCapture(input_video)
#cap = cv2.VideoCapture('./model/dataset/naranjas2/trimed_video.mp4')
#cap = cv2.VideoCapture(r'C:\Users\josec\Documents\Github\fruit-ripeness-classificator\videos\naranja_videos2\video_20240906_013859.h264')
cap = cv2.VideoCapture(r'C:\Users\josec\Documents\Github\fruit-ripeness-classificator\videos\naranja_videos2\video_20240906_015220.h264')

# Define the codec and create a VideoWriter object (output format: .avi with XVID codec)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("width, height and fps")
print(frame_width, frame_height, fps)

output_file = f'./output_video_{time.time()}.avi'  # Output file name
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Variables for tracking
objects_queue = deque()
x_min = 150  # Define a minimum x-center value to consider when an object leaves the frame
last_fruit_out = -1
fruit_out_delay = 13  # ms

# Start reading from frame 30
start_frame = 700

frame_number = 0

while frame_number < start_frame:
    ret, _ = cap.read()
    frame_number += 1

# Function to finalize the classification of an object
def finalize_class(object_data):
    weighted_sum = 0
    total_confidence = 0
    num_appearances = len(object_data)

    for obj_class, confidence in object_data:
        weighted_sum += obj_class * confidence
        total_confidence += confidence

    final_class = round(weighted_sum / total_confidence)
    average_confidence = total_confidence / num_appearances

    return final_class, average_confidence

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break  # Exit if video ends

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Creo que no debo hacer esto
    frame_number += 1  # Increment frame number
    
    # Mirror the frame (flip horizontally)
    img = cv2.flip(img, 1)
    
    if img is not None:
        """ INFERENCE """
        results = model(img)

        new_objects = []  # To store objects detected in the current frame
        
        for idx, result in enumerate(results):
            """ NMS """
            dets = result.boxes.data.cpu()
            boxes = dets[:, :4]  # x1, y1, x2, y2
            scores = dets[:, 4]  # confidence
            indices = nms(boxes, scores, iou_threshold=0.5)

            filtered_dets = dets[indices].numpy()
            boxes = filtered_dets[:, :4]  # x1, y1, x2, y2
            clss = filtered_dets[:, 5]  # object class
            confidences = filtered_dets[:, 4]  # confidence scores

            for box, obj_class, confidence in zip(boxes, clss, confidences):
                x_center = (box[0] + box[2]) / 2  # (x1 + x2) / 2 to get the x-center
                new_objects.append((x_center, obj_class, confidence))

        """ TRACKING """
        height, width, _ = img.shape
        exit_line = width - x_min

        new_objects.sort(key=lambda x: x[0])  # Sort by x_center

        for i, new_obj in enumerate(new_objects):
            x_center, obj_class, confidence = new_obj

            if i < len(objects_queue):
                # Update existing objects in the queue
                objects_queue[i] = (x_center, objects_queue[i][1])  # Update x_center
                objects_queue[i][1].append((obj_class, confidence))  # Update object history
            else:
                # Add new objects to the queue
                objects_queue.append((x_center, [(obj_class, confidence)]))


        time_since_last_out = time.time() - last_fruit_out
        
        """ DECISION AND TRACK FINALIZATION """
        for idx, tracked_obj in enumerate(list(objects_queue)):
            x_center = tracked_obj[0]
            
            print(f"{x_center} of {exit_line} and {time_since_last_out} of {fruit_out_delay}")

            if x_center > exit_line and (time_since_last_out > fruit_out_delay):
                print("\n---------------------------------------------------")
                print("Finalizing class")
                final_class, final_confidence = finalize_class(tracked_obj[1])
                print(f"Object {tracked_obj[0]} left the frame. Final class: {final_class}, Confidence: {final_confidence}")
                print("---------------------------------------------------")
                objects_queue.remove(tracked_obj)  # Remove the finalized object from the queue
                last_fruit_out = time.time()    

        """ ANNOTATION """
        annotator = Annotator(img, line_width=2)
        for box, cls, conf in zip(boxes, clss, confidences):
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(box, label=label, color=colors(int(cls)))

        # Draw a vertical line at x_min
        cv2.line(img, (exit_line, 0), (exit_line, height), (0, 255, 0), 2)  # Green line, 2 px thick
        
        # Annotate the time difference in the top-right corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        time_annotation = f"{time_since_last_out:.2f}/{fruit_out_delay:.2f} seconds"
        cv2.putText(img, time_annotation, (width - 400, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        
        # Annotate the frame number in the top-left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'Frame: {frame_number}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Saving frame for the video
        out.write(img)

        # Resize the frame to 720p resolution
        resized_img = cv2.resize(img, (1280, 720))  # Resize to 1280x720 (720p)

        # Display the resized frame
        #cv2.imshow('Inference + Tracking (720p)', resized_img)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
