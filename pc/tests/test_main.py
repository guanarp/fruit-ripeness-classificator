import socket
import cv2
import numpy as np
import struct
import time
import sys
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
import torch
from torchvision.ops import nms
from collections import deque


# Load the YOLO model and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not torch.cuda.is_available():
    print("Cuda not available :(")
#model = YOLO('yolov8n.pt')  # You can choose a different model based on your requirement
#model = YOLO(r'fruit-ripeness-classificator/model/best_exp5.pt')  # You can choose a different model based on your requirement
model = YOLO(r'fruit-ripeness-classificator/model/best_exp6.pt')  # You can choose a different model based on your requirement
#model = YOLO('yolov 9e.pt')
names = model.model.names

#HOST = '192.168.1.132'
HOST = '10.4.0.216' #varia
PORT = 9999
buffSize = 65000

RESPONSE_HOST = '192.168.135.30'
RESPONSE_PORT = 9997

FORWARD_HOST = '192.168.135.132'
FORWARD_PORT = 9998

# Create a UDP socket and bind it
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((HOST, PORT))
server.settimeout(2)  # Set a 2-second timeout for receiving data

forward_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
forward_server.connect((FORWARD_HOST,FORWARD_PORT))

response_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
response_server.connect((RESPONSE_HOST, RESPONSE_PORT))


print('Now waiting for frames...')

#cv2.startWindowThread()
counter = 0
sum_t = 0

# Modifications start here
MAX_DGRAM_SIZE = 65000
header_size = 5  # Length of 'FRAME' header
first_time = True
RESPONSE_HEADER = b'CLASS'
FORWARD_HEADER = b'FRAME'

objects_queue = deque()

x_min = 150  # Define a minimum x-center value to consider when an object leaves the frame

output_file = f'./output_video_{time.time()}.avi'  # Output file name
frame_width = 1280
frame_height = 720
fps = 12

# Define the codec and create a VideoWriter object (output format: .avi with XVID codec)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

last_fruit_out = -1
#fruit_out_delay = 330 #ms
fruit_out_delay = 330000000 #330M ns = 330 ms
#fruit_out_delay = 500000000
#fruit_out_delay = 550000000
#fruit_out_delay = 750000000

def finalize_class(object_data):
    # object_data is a list of tuples (class, confidence)
    weighted_sum = 0
    total_confidence = 0
    
    num_appearances = len(object_data)

    # Calculate weighted sum and total confidence
    for obj_class, confidence in object_data:
        weighted_sum += obj_class * confidence
        total_confidence += confidence

    # Compute the weighted average and round it
    final_class = round(weighted_sum / total_confidence)
    
    # Compute average confidence
    average_confidence = total_confidence / num_appearances
    
    return final_class, average_confidence  # Returning the final class and the total confidence



message_sent = None

while True:
    start = time.time_ns()
    start_receive = time.time_ns()
    chunks = []
    
    try:
        # Receive data with timeout handling
        data, address = server.recvfrom(buffSize)
    except socket.timeout:
        print("Socket timeout, waiting for more frames...")
        continue
    
    # Check for the unique header
    if data.startswith(b'FRAME'):
        # Extract the length of the image data (following the 5-byte header and 4-byte length)
        length = struct.unpack('I', data[header_size:header_size + 4])[0]
        # Extract the initial chunk of image data
        img_data = data[header_size + 4:]
        chunks.append(img_data)
        remaining = length - len(img_data)
        
        # Continue receiving the remaining chunks
        while remaining > 0:
            chunk, address = server.recvfrom(MAX_DGRAM_SIZE)
            chunks.append(chunk)
            remaining -= len(chunk)
        
    receive_time = (time.time_ns() - start_receive) / 1000000  # Convert to ms
    start_process = time.time_ns()

    # Reassemble the image data
    img_data = b''.join(chunks)
    if len(img_data) == length:
        # Convert data to numpy array and decode the image
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        process_time = (time.time_ns() - start_process) / 1000000  # Convert to ms
        start_display = time.time_ns()
        
    
        if img is not None:
            """ INFERENCE """
            results = model(img, verbose=False)

            # Track objects and maintain the queue
            new_objects = []  # To store objects detected in the current frame
            
            for idx, result in enumerate(results):
                """ NMS """
                dets = result.boxes.data.cpu()
                boxes = dets[:, :4]  # x1, y1, x2, y2
                scores = dets[:, 4]  # confidence
                indices = nms(boxes, scores, iou_threshold=0.5)

                # Filter for best candidates
                filtered_dets = dets[indices].numpy()
                #print(filtered_dets)
                boxes = filtered_dets[:, :4]  # x1, y1, x2, y2
                clss = filtered_dets[:, 5]  # object class
                confidences = filtered_dets[:, 4]  # confidence scores
                #print("Filtered dets:", filtered_dets)

                # Calculate x_center for each object and store with class and confidence
                for box, obj_class, confidence in zip(boxes, clss, confidences):
                    x_center = (box[0] + box[2]) / 2  # (x1 + x2) / 2 to get the x-center
                    if x_center < 1194:
                        new_objects.append((x_center, obj_class, confidence))
                    else:
                        print(f"IGNORING because the x_center is {x_center}")

            """ TRACKING """
            # Get the dimensions of the frame (height and width)
            height, width, _ = img.shape
            exit_line = width - x_min
            
            # Sort new objects and queue based on x_center (left to right)
            new_objects.sort(key=lambda x: x[0])  # Sort by x_center (0th element)

            # #Ahora agregue que ordene el objects queue tambien
            # objects_queue.sort(key=lambda x: x[0])
            # Tienen que estar ordenados al reves porque el de x mas chica en new_objects debe ser el ult objeto de objects_queue

            # Iterate over new_objects and update the queue accordingly
            for i, new_obj in enumerate(new_objects):
                x_center, obj_class, confidence = new_obj
        
                if i < len(objects_queue):
                    # Update existing objects in the queue
                    # Append new x_center to the list (store multiple x_centers)
                    objects_queue[i][0].append(x_center)  # Append to the list of x_center
                    objects_queue[i][1].append((obj_class, confidence))  # Update object history with class/confidence
                else:
                    # If new_objects has more items, add new ones to the queue
                    objects_queue.append(([x_center], [(obj_class, confidence)]))

            time_since_last_out = time.time_ns() - last_fruit_out
            
            """ DECISION Y RESPUESTA """
            # Now, check if any objects in the queue need to be finalized
            print(list(objects_queue))
            for idx, tracked_obj in enumerate(list(objects_queue)):  # Convert to list to safely remove items
                x_center = tracked_obj[0][-1] # Get the last x_center from the list
        
                if x_center > exit_line and (time_since_last_out) > fruit_out_delay:
                    # The object has moved out of the frame, finalize and remove it
                    final_class, final_confidence = finalize_class(tracked_obj[1])
                    print("--------------------------------------------------------------------------")
                    print(f"Object {tracked_obj[0]} left the frame. Final class: {final_class}, Confidence: {final_confidence}")
                    objects_queue.remove(tracked_obj)  # Remove the finalized object from the queue
                    
                    print(f"Final class: {final_class}")
                    full_message = RESPONSE_HEADER + str(final_class).encode('utf-8')
                    
                    # Send message encoded as bytes
                    server.sendto(full_message, (RESPONSE_HOST, RESPONSE_PORT))
                    print(f"Message sent: {full_message} at time: {time.time_ns()/1000000:0f}")
                    print("--------------------------------------------------------------------------")
                    last_fruit_out = time.time_ns()
                    message_sent = final_class
                    full_message = None

            """ ANNOTATION """
            annotator = Annotator(img, line_width=2)
            # Render detections on the image
            for box, cls, conf in zip(boxes, clss, confidences):
                label = f"{names[int(cls)]} {conf:.2f}"  # Format the label with class name and confidence
                annotator.box_label(box, label=label, color=colors(int(cls)))
        
            # Draw a vertical line at x_min (from top to bottom)
            cv2.line(img, (exit_line, 0), (exit_line, height), (0, 255, 0), 2)  # Green line, 2 px thick

            # Annotate the time difference in the top-right corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            time_annotation = f"{time_since_last_out/1000000:.2f}/{fruit_out_delay/1000000:.2f} ms"
            cv2.putText(img, time_annotation, (width - 400, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #Message sent
            # if message_sent is not None:
            #     cv2.putText(img, message_sent, (width//2, height//2), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #     message_sent = None

            # Se supone que ya no debo hacer plot()
            # Apply the detections on the image
            # img_with_detections = results[0].plot()

            """ DISPLAY O CAPTURA O STREAMING """
            # Encode the image
            result, imgencode = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            data = imgencode.tobytes()
            data_len = len(data)
            #print("data lenght:",data_len)
            length = struct.pack('I', data_len)  # Image data length as unsigned int
            message = FORWARD_HEADER + length #+ data

            #print(f"Forwarding image with inference to {FORWARD_HOST}, {FORWARD_PORT}")
            # Send the byte length of the encoded image data
            forward_server.sendto(message, (FORWARD_HOST, FORWARD_PORT))

            # Saving frame for the video
            out.write(img)

            
            #Display the image
            #cv2.imshow('frames', img_with_detections)
            #cv2.imshow('frames',img)
            #imS = cv2.resize(img, (1280, 720))     
            #cv2.imshow('frames', img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

            """ OTRO """
            
            
            
            
            # # Measure frame time and print average
            # end = time.time_ns()
            # delta_t = (end - start) / 1000000  # Convert ns to ms
            # sum_t += delta_t
            # counter += 1
            # avg_t = sum_t / counter
            
            # print(f"\rCurr frame time: {delta_t:.2f} ms", end="\t")
            # print(f"Avg time: {avg_t:.2f} ms")
            #sys.stdout.flush()  # Make sure to flush the output to update it in real-time
        display_time = (time.time_ns() - start_display) / 1000000  # Convert to ms
        total_time = (time.time_ns() - start) / 1000000  # Convert to ms

        if not(first_time):
            sum_t += total_time
            counter += 1
            avg_t = sum_t / counter
            # print(f"\rReceive time: {receive_time:.2f} ms", end="\t")
            # print(f"Process time: {process_time:.2f} ms", end="\t")
            # print(f"Display time: {display_time:.2f} ms", end="\t")
            # print(f"Total time: {total_time:.2f} ms", end="\t")
            # print(f"Avg time: {avg_t:.2f} ms")
            sys.stdout.flush()
        else:
            first_time = False
    else:
        print("Incomplete frame data received")
        print(f"img data {len(img_data)}, length {length}")
# Modifications end here

# Cleanup
server.close()
cv2.destroyAllWindows()
out.release()


