import cv2
import numpy as np
from picamera2 import Picamera2
import socket
import struct
import multiprocessing as mp
import mmap
from ctypes import CDLL, c_int, c_long, get_errno
import os
from threading import Thread
from time import sleep, time, time_ns

class StreamingProcess(mp.Process):
    """A separate process for streaming frames received from Picamera2."""

    def __init__(self, picam2, name='main', host='192.168.1.132', port=9999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[StreamingProcess __init__] Initializing the streaming process.")
        self.config = picam2.camera_configuration()[name]
        self._picam2_pid = os.getpid()
        self._send_queue = mp.Queue()
        self.host = host
        self.port = port
        self._syscall = CDLL(None, use_errno=True).syscall
        self._syscall.argtypes = [c_long]
        self._stream = picam2.stream_map[name]
        # Note: Starting the process is now moved outside of __init__

    def _format_array(self, mem):
        array = np.array(mem, copy=False, dtype=np.uint8)
        width, height = self.config['size']
        stride = self.config['stride']
        fmt = self.config['format']
        if fmt == 'YUV420':
            return array.reshape((height + height // 2, stride))
        array = array.reshape((height, stride))
        if fmt in ('RGB888', 'BGR888'):
            return array[:, :width * 3].reshape((height, width, 3))
        elif fmt in ("XBGR8888", "XRGB8888"):
            return array[:, :width * 4].reshape((height, width, 4))
        return array

    def _map_fd(self, picam2_fd):
        #print(f"[StreamingProcess _map_fd] Mapping fd {picam2_fd} from pid {self._picam2_pid}")
        PIDFD_OPEN = 434  # System call number for pidfd_open
        PIDFD_GETFD = 438  # System call number for pidfd_getfd

        # Open a file descriptor referring to the target process
        pidfd = self._syscall(PIDFD_OPEN, c_int(self._picam2_pid), c_int(0))
        if pidfd == -1:
            errno = get_errno()
            raise OSError(errno, os.strerror(errno), "pidfd_open syscall failed")

        # Obtain a duplicate of the target file descriptor
        fd = self._syscall(PIDFD_GETFD, c_int(pidfd), c_int(picam2_fd), c_int(0))
        if fd == -1:
            errno = get_errno()
            raise OSError(errno, os.strerror(errno), "pidfd_getfd syscall failed")
        return fd

    def capture_shared_array(self):
        #print("[StreamingProcess capture_shared_array] Waiting to receive data in process...")
        msg = self._send_queue.get()
        if msg == "TERMINATE":
            print("[StreamingProcess capture_shared_array] Received TERMINATE signal.")
            return None
        picam2_fd, length = msg
        fd = self._map_fd(picam2_fd)
        mem = mmap.mmap(fd, length, mmap.MAP_SHARED, mmap.PROT_READ)
        array = self._format_array(mem)
        # It's good practice to close the duplicated file descriptor
        os.close(fd)
        return array

    def run(self):
        print("[StreamingProcess run] Streaming process started.")
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            server.connect((self.host, self.port))
            print(f"[StreamingProcess run] Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"[StreamingProcess run] Failed to connect to {self.host}:{self.port} - {e}")
            return

        header = b'FRAME'
        MAX_DGRAM_SIZE = 65000

        while True:
            frame = self.capture_shared_array()
            if frame is None:
                print("[StreamingProcess run] No frame captured, terminating.")
                break
            encode_start = time_ns()
            result, imgencode = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]) #con 45 son 50 ms, con 80 tambien
            encode_time = (time_ns() - encode_start) / 1_000_000
            print(f"[StreamingProcess] Image encoding took: {encode_time:.4f} ms")
            if not result:
                print("[StreamingProcess run] Failed to encode frame, skipping.")
                continue
            data = imgencode.tobytes()
            #data = frame.tobytes()
            data_len = len(data)
            length = struct.pack('I', data_len)
            message = header + length
            send_start = time_ns()
            try:
                server.sendto(message, (self.host, self.port))

                for i in range(0, data_len, MAX_DGRAM_SIZE):
                    chunk = data[i:i + MAX_DGRAM_SIZE]
                    server.sendto(chunk, (self.host, self.port))

                #print('[StreamingProcess run] Sent one frame.')
            except Exception as e:
                print(f"[StreamingProcess run] Failed to send frame: {e}")
            send_time = (time_ns() - send_start) / 1_000_000
            print(f"[StreamingProcess] Sending frame took: {send_time:.4f} ms for length {data_len}\n")

        server.close()
        print("[StreamingProcess run] Socket closed, streaming process exiting.")

    def send_frame(self, request):
        plane = request.request.buffers[self._stream].planes[0]
        fd = plane.fd
        length = plane.length
        #print("[StreamingProcess send_frame] Sending frame to the process.")
        self._send_queue.put((fd, length))

    def stop(self):
        #print("[StreamingProcess stop] Sending TERMINATE signal to the process.")
        self._send_queue.put("TERMINATE")
        self.join()
        #print("[StreamingProcess stop] Streaming process has been joined and stopped.")

if __name__ == "__main__":
    print("[Main] Starting main process.")
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size":(1280,720), "format":"BGR888"},controls={'FrameRate': 24}, display="main") #
    #10 se ve muyy lindo pero con ghosting
    #24 esta ok
    picam2.configure(config)
    picam2.start()
    print("[Main] Picamera2 has been started.")

    process = StreamingProcess(picam2, 'main')
    print("[Main] StreamingProcess instance created.")
    process.start()
    print("[Main] StreamingProcess has been started.")

    try:
        print("[Main] Beginning to capture and send frames.")
        while True:  # Continuous streaming loop
            start_time = time()
            request = picam2.capture_request()  # Capture a request without using 'with'
            capture_time = time() - start_time
            print(f"[Main] Starting capture request. completed in {capture_time:.4f}")
            start_time = time()
            process.send_frame(request)
            send_frame_time = time() - start_time
            #print(f"Send frame in {send_frame_time:.4f}")
            request.release()  # Manually release the request after sending
    except KeyboardInterrupt:
        print("[Main] Streaming interrupted by user.")
    finally:
        process.stop()
        print("[Main] Streaming process has been stopped.")
        picam2.close()
        print("[Main] Picamera2 has been closed.")
