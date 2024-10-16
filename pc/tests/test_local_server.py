import socket
import cv2
import numpy as np
import struct
import time
import sys


#HOST = '192.168.1.132'
#HOST = '192.168.135.31' #varia
#HOST = '200.10.231.216'
#HOST = '0.0.0.0'
#HOST = 'localhost'
HOST = '10.4.0.216' #Ip local
PORT = 9999
buffSize = 65000

# Create a UDP socket and bind it
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((HOST, PORT))
server.settimeout(2)  # Set a 2-second timeout for receiving data
print('Now waiting for frames...')

cv2.startWindowThread()
counter = 0
sum_t = 0

# Modifications start here
MAX_DGRAM_SIZE = 65000
header_size = 5  # Length of 'FRAME' header
first_time = True


output_file = './output_video.avi'  # Output file name
frame_width = 1280
frame_height = 720
fps = 60

# Define the codec and create a VideoWriter object (output format: .avi with XVID codec)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

try:
    while True:
        start = time.time_ns()
        start_receive = time.time_ns()
        chunks = []
        while True:
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
    
                break
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
            # Display the image
            if img is not None:
                # results = model(img)
                # for idx, result in enumerate(results):
                #     dets = result.boxes.data.cpu()
                #     boxes = dets[:, :4]  # x1, y1, x2, y2
                #     scores = dets[:, 4]  # confidence
                #     indices = nms(boxes, scores, iou_threshold=0.5)
                #     filtered_dets = dets[indices].numpy()
                #     #print(filtered_dets)
                #     boxes = filtered_dets[:,:4]
                #     clss = filtered_dets[:,5]
                #     print("Filtered dets:", filtered_dets)
                # # Perform detection
                # # boxes = results[0].boxes.xyxy.cpu()
                # # clss = results[0].boxes.cls.cpu().tolist()
                # # #print(results)
                # annotator = Annotator(img, line_width=2)
                # # Render detections on the image
                # for box, cls in zip(boxes, clss):
                #     annotator.box_label(box, label=names[int(cls)], color=colors(int(cls)))
                    
                # #results.render()  # Apply the detections on the image
                # img_with_detections = results[0].plot()
    
                #Display the image
                # cv2.imshow('frames', img_with_detections)
                #cv2.imshow('frames',img)
                out.write(img)
                #imS = cv2.resize(img, (1280, 720))     
                #cv2.imshow('frames', img)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                
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
                print(f"\rReceive time: {receive_time:.2f} ms", end="\t")
                print(f"Process time: {process_time:.2f} ms", end="\t")
                print(f"Display time: {display_time:.2f} ms", end="\t")
                print(f"Total time: {total_time:.2f} ms", end="\t")
                print(f"Avg time: {avg_t:.2f} ms")
                sys.stdout.flush()
            else:
                first_time = False
        else:
            print("Incomplete frame data received")
            print(f"img data {len(img_data)}, expected length {length}")
            print([len(x) for x in chunks])
# Modifications end here
except Exception as e:
    print(e)
finally:
    out.release()
    
    # Cleanup
    server.close()
    cv2.destroyAllWindows()
