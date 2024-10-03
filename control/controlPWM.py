import RPi.GPIO as GPIO
from time import sleep
import time
import random

## add your servo BOARD PIN number ##
SERVOA_PIN = 12 # Use the correct GPIO pin number where your servo is connected
SERVOB_PIN = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVOA_PIN, GPIO.OUT)
GPIO.setup(SERVOB_PIN, GPIO.OUT)

pwma=GPIO.PWM(SERVOA_PIN, 50)
pwma.start(0)

pwmb=GPIO.PWM(SERVOB_PIN, 50)
pwmb.start(0)

def set_servo_angle(pwm, angle):
    # Calculate duty cycle for the given angle (angle between 0 and 180)
    duty_cycle = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # Allow time for the servo to move to the position
    pwm.ChangeDutyCycle(0)  # Stop sending PWM signal after movement

print("begin test")


exit_time = 2
last_movement_time = time.time()

try:
	while True:
		if time.time() - last_movement_time >= exit_time:
			number = random.choice([0,1,2])
			print(number,"\n")
			if number == 1:
				set_servo_angle(pwma,60)
				set_servo_angle(pwmb,60)
				time.sleep(1)
				set_servo_angle(pwma,90)
				set_servo_angle(pwmb,90)
			elif number == 2:
				set_servo_angle(pwma,120)
				set_servo_angle(pwmb,120)
				time.sleep(1)
				set_servo_angle(pwma,90)
				set_servo_angle(pwmb,90)
			last_movement_time = time.time()
			
			
			
	
except KeyboardInterrupt:
	print("Program stopped by user")
	
finally:
	pwma.stop()
	pwmb.stop()
	GPIO.cleanup()  # Reset GPIO settings when the program is terminated
