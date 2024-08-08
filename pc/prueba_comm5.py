import socket
import cv2
import numpy as np
import struct
import time
import sys

HOST = '192.168.1.132'
PORT = 9999
buffSize = 65535

# Create a UDP socket
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Bind the address and port
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
            end = time.time_ns()
            cv2.imshow('frames', img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
            delta_t = (end - start)/1000000
            sum_t += delta_t
            counter += 1
            avg_t = sum_t / counter
            
            print(f"\r\rCurr frame time: {delta_t} ms", end="\t")
            print(f"Avg time: {avg_t} ms")

            sys.stdout.write('\033[A')
            sys.stdout.write('\033[K')
        else:
            print("Incomplete frame data received")
    else:
        print("Unrecognized data received")

# Cleanup
server.close()
cv2.destroyAllWindows()
