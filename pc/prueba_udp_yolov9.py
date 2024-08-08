import socket

import cv2
import numpy as np
import struct
import time
import sys
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator

import torch

HOST = '192.168.1.132'
PORT = 9999
buffSize = 65535

# Load the YOLO model and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not torch.cuda.is_available():
    raise ValueError("Cuda not available")
#model = YOLO('yolov8n.pt')  # You can choose a different model based on your requirement
model = YOLO('yolov9c.pt')  # You can choose a different model based on your requirement
#model = YOLO('yolov 9e.pt')
names = model.model.names

# Create a UDP socket and bind it
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((HOST, PORT))
print('Now waiting for frames...')

cv2.startWindowThread()
counter = 0
sum_t = 0
while True:
    start = time.time_ns()
    # Receive data
    data, address = server.recvfrom(buffSize)
    # Check for the unique header
    if data.startswith(b'FRAME'):
        # Extract the length of the image data (following the 5-byte header and 4-byte length)
        length = struct.unpack('I', data[5:9])[0]
        # Extract the image data
        img_data = data[9:9+length]
        # Make sure the length of the received data matches the expected length
        if len(img_data) == length:
            # Convert data to numpy array and decode the image
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Display the image
            if img is not None:
                # Perform detection
                results = model(img)
                boxes = results[0].boxes.xyxy.cpu()
                clss = results[0].boxes.cls.cpu().tolist()
                #print(results)
                annotator = Annotator(img, line_width=2)
                # Render detections on the image
                for box, cls in zip(boxes, clss):
                    annotator.box_label(box, label=names[int(cls)], color=colors(int(cls)))
                    
                #results.render()  # Apply the detections on the image
                #img_with_detections = results.imgs[0]

                # Display the image
                #cv2.imshow('frames', img_with_detections)
                cv2.imshow('frames', img)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                
                # Measure frame time and print average
                end = time.time_ns()
                delta_t = (end - start) / 1000000  # Convert ns to ms
                sum_t += delta_t
                counter += 1
                avg_t = sum_t / counter
                
                print(f"\rCurr frame time: {delta_t:.2f} ms", end="\t")
                print(f"Avg time: {avg_t:.2f} ms")
                sys.stdout.flush()  # Make sure to flush the output to update it in real-time
        else:
            print("Incomplete frame data received")
    else:
        print("Unrecognized data received")

# Cleanup
server.close()
cv2.destroyAllWindows()
