import RPi.GPIO as GPIO
import time
import random

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

try:
	while True:
		sensor_state = GPIO.input(SENSOR_PIN)
        
		if sensor_state == GPIO.LOW and previous_state == GPIO.HIGH:
			print("Reading an object")
			last_trigger_time = time.time()  # Update the last trigger time
			GPIO.output(FREQ_PIN, GPIO.HIGH)
			time.sleep(0.1)
			GPIO.output(MOTOR_PIN, GPIO.HIGH)  # Turn on output
			fruits.append(last_trigger_time)
            
		else:
			print(f"didn't detect\t {time.time()  - last_trigger_time}")
        
		previous_state = sensor_state    
		print(f"Number of fruits {len(fruits)}")    
		if fruits and (time.time() - fruits[0] >= exit_time):
			print("Dropping fruit")
			number = random.choice([1,2])
			if number == 1:
				set_servo_angle(pwma,60)
				set_servo_angle(pwmb,60)
				time.sleep(0.5)
				set_servo_angle(pwma,90)
				set_servo_angle(pwmb,90)
			elif number == 2:
				set_servo_angle(pwma,120)
				set_servo_angle(pwmb,120)
				time.sleep(0.5)
				set_servo_angle(pwma,90)
				set_servo_angle(pwmb,90)
			fruits.pop()
				
        
			# Check if 20 seconds have passed since the last sensor trigger
			if time.time() - last_trigger_time > output_on_time:
				print("Didnt read an object in the last 20s")
				GPIO.output(MOTOR_PIN, GPIO.LOW)  # Turn off output
				GPIO.output(FREQ_PIN, GPIO.LOW)

			time.sleep(0.1)  # Small delay to avoid CPU overload

except KeyboardInterrupt:
	print("Program stopped by user")
    
finally:
	pwma.stop()
	pwmb.stop()
	GPIO.cleanup()  # Reset GPIO settings when the program is terminated

