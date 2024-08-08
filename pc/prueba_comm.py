import cv2
import socket
import threading

def udp_server(host='192.168.1.132', port=1935):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((host, port))
    print(f"UDP server up and listening on {host}:{port}")
    while True:
        data, address = server_socket.recvfrom(65535) #max value for UDP
        #print(f"Message from {address}: {data.decode()}")

def stream_capture(stream_url):
    cap = cv2.VideoCapture(stream_url)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    udp_thread = threading.Thread(target=udp_server)
    stream_thread = threading.Thread(target=stream_capture, args=('udp://192.168.1.135:1935',))

    udp_thread.start()
    stream_thread.start()

    udp_thread.join()
    stream_thread.join()
