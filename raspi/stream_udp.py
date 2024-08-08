import cv2
import numpy as np
from picamera2 import Picamera2
import socket
import struct
from time import sleep

picam2 = Picamera2()
picam2.set_controls({"FrameRate":60})
print(picam2.sensor_modes)
#while True:
#    continue
#config = picam2.create_still_configuration({"size":(640,480), "format" : "XRGB8888"})
#picam2.configure(config)
#picam2.framerate=30
##print(picam2.sensor_modes)

try:
    picam2.start()
    sleep(2)
except Exception as e:
    print(f"failed to start the camera: {e}")
    sys.exit(1)

#cam = cv2.VideoCapture(1)

# Assuming these are your current streaming settings
HOST = '192.168.1.132'  # The IP address of your PC
PORT = 9999  # The port used for the connection

# Set up a socket for sending data (streaming images)
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.connect((HOST,PORT)) # Connect to target address
header = b'FRAME'  # Unique header

print('Now starting to send frames...')

MAX_DGRAM_SIZE = 65000 #slightly less than 65507

# Optionally, set up a separate socket for receiving responses, if needed
#client_socket.bind(('0.0.0.0', 8888))  # Listen on a different port

#capture = cv2.VideoCapture(1, cv2.CAP_V4L2)

try:
    
    while True:
        #print("FIRST FRAME")
        #success, frame = cam.read()
        frame = picam2.capture_array()
        if frame is None:
            print("Could not capture the frame")
            print(frame)
            continue  # If the frame was not captured successfully, try again

        # Encode the image
        result, imgencode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        data = imgencode.tobytes()
        data_len = len(data)
        print(data_len)
        length = struct.pack('I', data_len)  # Image data length as unsigned int
        message = header + length #+ data

        # Send the byte length of the encoded image data
        server.sendto(message, (HOST, PORT))
        
        print('Have sent size')
        
        for i in range(0,data_len, MAX_DGRAM_SIZE):
            chunk = data[i:i+MAX_DGRAM_SIZE]
            server.sendto(chunk, (HOST, PORT))
            
        print('Have sent one frame')

except Exception as e:
    print(e)
finally:
    # Send a close message
    server.sendto(struct.pack('B', 1), (HOST, PORT))
    
    # Release camera resources
    # capture.release()
    
    # Close the socket
    server.close()
