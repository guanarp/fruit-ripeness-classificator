import cv2
import socket
import threading
import queue
import numpy as np

# Initialize a queue for frame storage
frameQueue = queue.Queue(maxsize=10)  # Adjust maxsize as needed

host='192.168.1.132'
port=5000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((host, port))
print(f"UDP server up and listening on {host}:{port}")

def udp_server(host='192.168.1.132', port=5000):
    while True:
        data, _ = server_socket.recvfrom(65535)  # Receive data
        #print(data)
        nparr = np.frombuffer(data, np.uint8)  # Convert data to a numpy array
        #print(nparr)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode image
        print(frame)
        if frame is not None:
            if not frameQueue.full():
                frameQueue.put(frame)

def stream_capture(stream_url):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Unable to open video stream.")
        return  # Exit the function if we can't open the stream

    while True:
        ret, frame = cap.read()
        if ret:
            print("Captured a frame")  # Diagnostic print
            if not frameQueue.full():
                frameQueue.put(frame)
        else:
            print("Error: Unable to read from the video stream.")
            break
    cap.release()

def display_frames():
    print("Display frames")
    #while True:
    print(frameQueue.empty())
    if not frameQueue.empty():
        frame = frameQueue.get()
        cv2.imshow('Frame', frame)
        print("imshow")
        #if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' key press
            #break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    udp_thread = threading.Thread(target=udp_server)
    #stream_thread = threading.Thread(target=stream_capture, args=('udp://192.168.1.135:5000',))  # Update with your URL

    udp_thread.start()
    #stream_thread.start()
    udp_thread.join()
    while True:
        display_frames()  # Run in the main thread

        
    #stream_thread.join()
