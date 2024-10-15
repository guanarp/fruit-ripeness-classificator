import RPi.GPIO as GPIO
import time
import random
import asyncio
import socket

import time
import struct
from picamera2 import Picamera2
import cv2
import numpy as np

import os
from multiprocessing import Process, Queue
import mmap
from ctypes import CDLL, c_int, c_long, get_errno
from time import time, time_ns, sleep

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size":(1280,720), "format":"BGR888"},controls={'FrameRate': 24}, display="main") #
#10 se ve muyy lindo pero con ghosting
#24 esta ok
picam2.configure(config)
picam2.start()
sleep(1)



class StreamingProcess(Process):
    def __init__(self, picam2, name='main', host='192.168.1.132', port=9999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = picam2.camera_configuration()[name]
        self.host = host
        self.port = port
        self._send_queue = Queue()
        self._picam2_pid = os.getpid()
        self._syscall = CDLL(None, use_errno=True).syscall
        self._syscall.argtypes = [c_long]
        self._stream = picam2.stream_map[name]

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
        return array

    def _map_fd(self, picam2_fd):
        PIDFD_OPEN = 434
        PIDFD_GETFD = 438
        pidfd = self._syscall(PIDFD_OPEN, c_int(self._picam2_pid), c_int(0))
        fd = self._syscall(PIDFD_GETFD, c_int(pidfd), c_int(picam2_fd), c_int(0))
        return fd

    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.connect((self.host, self.port))

        header = b'FRAME'
        MAX_DGRAM_SIZE = 65000

        while True:
            picam2_fd, length = self._send_queue.get()
            fd = self._map_fd(picam2_fd)
            mem = mmap.mmap(fd, length, mmap.MAP_SHARED, mmap.PROT_READ)
            frame = self._format_array(mem)
            os.close(fd)

            result, imgencode = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            data = imgencode.tobytes()
            data_len = len(data)
            length = struct.pack('I', data_len)
            message = header + length

            server.sendto(message, (self.host, self.port))
            for i in range(0, data_len, MAX_DGRAM_SIZE):
                chunk = data[i:i + MAX_DGRAM_SIZE]
                server.sendto(chunk, (self.host, self.port))

        server.close()

    def send_frame(self, request):
        plane = request.request.buffers[self._stream].planes[0]
        fd = plane.fd
        length = plane.length
        #print("[StreamingProcess send_frame] Sending frame to the process.")
        self._send_queue.put((fd, length))

    def stop(self):
        self.terminate()
        self.join()

GPIO.setmode(GPIO.BOARD)

# Define pins
SENSOR_PIN = 23  # Pin for the sensor input (adjust to your setup)
MOTOR_PIN = 32  # Pin for the digital output (adjust to your setup)
FREQ_PIN = 36

SERVOA_PIN = 12 # Use the correct GPIO pin number where your servo is connected
SERVOB_PIN = 33

GPIO.setup(SERVOA_PIN, GPIO.OUT)
GPIO.setup(SERVOB_PIN, GPIO.OUT)

pwma = GPIO.PWM(SERVOA_PIN, 50)
pwma.start(0)

pwmb = GPIO.PWM(SERVOB_PIN, 50)
pwmb.start(0)

def set_servo_angle(pwm, angle):
	# Calculate duty cycle for the given angle (angle between 0 and 180)
	duty_cycle = 2 + (angle / 18)
	pwm.ChangeDutyCycle(duty_cycle)
	time.sleep(0.5)  # Allow time for the servo to move to the position
	pwm.ChangeDutyCycle(0)  # Stop sending PWM signal after movement


# Setup GPIO mode
GPIO.setup(SENSOR_PIN, GPIO.IN)  # Sensor input
GPIO.setup(MOTOR_PIN, GPIO.OUT)  # Output control
GPIO.setup(FREQ_PIN, GPIO.OUT)  # Output control
 
# Variables
output_on_time = 6  # Output stays on for 20 seconds
last_trigger_time = 0  # To track when the sensor was last triggered
exit_time = 2

fruits = []

previous_state = GPIO.HIGH

udp_data = None

# Set up asyncio event loop
loop = asyncio.get_event_loop()

# UDP setup using asyncio
UDP_IP = '192.168.135.30'  # Listen on all available interfaces
UDP_PORT = 9998      # Port to listen on

#UDP_IMAGE_IP = '10.4.0.216'
UDP_IMAGE_IP = '192.168.135.31' # Para verificar en mi notebook
UDP_IMAGE_PORT = 9999  # Port to send images
MAX_DGRAM_SIZE = 65000  # Slightly less than 65507 (UDP max size)
HEADER = b'FRAME'  # Frame header

async def udp_receiver():
    global udp_data
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)  # Make the socket non-blocking for asyncio

    print("-----------------------------------")
    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")
    
    while True:
        try:
            data, addr = await loop.run_in_executor(None, sock.recvfrom, 6)
            if data.startswith(b'FRAME'):
                udp_data = data[5:]
                print(f"Received message: {udp_data} from {addr}")
        except BlockingIOError:
            await asyncio.sleep(0.01)
            
async def gpio_handler():
	global last_trigger_time, previous_state, fruits, udp_data
    
	while True:
		# Sensor check
		sensor_state = GPIO.input(SENSOR_PIN)

		if sensor_state == GPIO.LOW and previous_state == GPIO.HIGH:
			print("Reading an object")
			last_trigger_time = time.time()  # Update the last trigger time
			GPIO.output(FREQ_PIN, GPIO.HIGH)
			time.sleep(0.1)
			GPIO.output(MOTOR_PIN, GPIO.HIGH)  # Turn on output
			fruits.append(last_trigger_time)
		else:
			print(f"Didn't detect\t {time.time() - last_trigger_time}")

		previous_state = sensor_state
		print(f"Number of fruits {len(fruits)}")

		if udp_data is None:
			print("------------------------------------")
			print("Waiting data")
			print("----------------------------------")
		else:
			print("------------------------------------")
			print(f"Received this data: {udp_data}")
			print("----------------------------------")

		if fruits and (time.time() - fruits[0] >= exit_time): #Esto tengo que reemplazar por la logica de que sale de imagen
			print("Dropping fruit")
            
			if udp_data == b"1":
				print("Action: Turning servos for dropping type 1")
				set_servo_angle(pwma, 60)
				set_servo_angle(pwmb, 60)
				time.sleep(0.5)
				set_servo_angle(pwma, 90)
				set_servo_angle(pwmb, 90)
				fruits.pop()  # Remove the processed fruit from the list
				udp_data = None
			elif udp_data == b"2":
				print("Action: Turning servos for dropping type 2")
				set_servo_angle(pwma, 120)
				set_servo_angle(pwmb, 120)
				time.sleep(0.5)
				set_servo_angle(pwma, 90)
				set_servo_angle(pwmb, 90)
				fruits.pop()  # Remove the processed fruit from the list
				udp_data = None
			else:
				print("No valid UDP action received, not dropping fruit")

			

			# Check if 6 seconds have passed since the last sensor trigger
			if time.time() - last_trigger_time > output_on_time:
				print("Didn't read an object in the last 6s")
				GPIO.output(MOTOR_PIN, GPIO.LOW)  # Turn off output
				GPIO.output(FREQ_PIN, GPIO.LOW)

		await asyncio.sleep(0.1)  # Small delay to avoid CPU overload

async def capture_and_stream():
	global picam2
	print("My cam is",picam2)
	process = StreamingProcess(picam2, 'main', host=UDP_IMAGE_IP, port=UDP_IMAGE_PORT)
	process.start()

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


try:
    # Run both UDP and GPIO handler concurrently
    #loop.run_until_complete(asyncio.gather(udp_receiver(), gpio_handler(), capture_and_stream()))
    loop.run_until_complete(asyncio.gather(capture_and_stream()))

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    pwma.stop()
    pwmb.stop()
    GPIO.cleanup()  # Reset GPIO settings when the program is terminated
    loop.close()  # Close the asyncio loop
    picam2.close()
