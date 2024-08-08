import cv2
import numpy as np
from picamera2 import Picamera2
#from picamera.array import PiRGBArray
import time


picam2 = Picamera2()

#preview_config = picam2.create_preview_configuration(main={"size": (640, 480), "format" : "XRGB8888"})
#config = picam2.create_preview_configuration({})
config = picam2.create_video_configuration({"format" : "XRGB8888"}) 
picam2.configure(config)


# Configure the camera for video capture
#video_config = picam2.create_video_configuration(main={"size": (640, 480), "format" : "XRGB8888"})
#picam2.configure(video_config)

picam2.start()
#cam = cv2.VideoCapture(1)

# Capture frames from the camera
while True:
    im = picam2.capture_array()
    #success, im = cam.read()

    # Display the image on screen
    cv2.imshow("Frame", im)

    # If the 'q' key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
camera.close()

