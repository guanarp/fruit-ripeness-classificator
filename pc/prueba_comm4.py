import cv2
import numpy
import socket
import struct
import time
import sys
HOST = '192.168.1.132'
PORT = 9999
buffSize = 65535
# Create a UDP socket
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #averiguar los params que son
# Bind the address and port
server.bind((HOST, PORT))
#server.listen(0) pa que?
# Accept a single connection and make a file-like object out of it
#connection = server.accept()[0].makefile('rb')
print('Now waiting for frames...')
cv2.startWindowThread()
counter = 0
sum_t = 0
while True:
    start = time.time_ns()
    # First, receive the byte length
    data, address = server.recvfrom(buffSize)
    #print("After receiving")
    # If receiving a close message, stop the program
    if len(data) == 1 and data[0] == 1:
        print("Closing server")
        server.close()
        cv2.destroyAllWindows()
        exit()
    # Perform a simple check; the length value is of type int and takes up four bytes
    if len(data) != 4:
        length = 0
    else:
        length = struct.unpack('i', data)[0]  # Length value
    # Receive encoded image data
    data, address = server.recvfrom(buffSize)
    # Perform a simple check
    if length != len(data):
        print("Skipping\n")
        print(length,len(data))
        continue
    # Format conversion
    data = numpy.array(bytearray(data))
    # Decode the image
    imgdecode = cv2.imdecode(data, 1)
    #print('Have received one frame')
    # Display the frame in a window
    cv2.imshow('frames', imgdecode)
    # Press "ESC" to exit
    end = time.time_ns()
    if cv2.waitKey(1) & 0xFF == 27:
        break
    delta_t = (end - start)/1000000
    sum_t += delta_t
    counter += 1
    avg_t = sum_t / counter
    
    print(f"\r\rCurr frame time: {delta_t} ms", end="\t")
    print(f"Avg time: {avg_t} ms")

    sys.stdout.write('\033[A')
    sys.stdout.write('\033[K')
# Close the socket and window
server.close()
cv2.destroyAllWindows()