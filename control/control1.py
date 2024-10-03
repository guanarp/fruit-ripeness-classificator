import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

led_pin = 18
GPIO.setup(led_pin, GPIO.OUT)

try:
	while True:
		GPIO.output(led_pin, GPIO.HIGH)
		print("Led encendido")
		time.sleep(1)
		
		GPIO.output(led_pin, GPIO.LOW)
		print("Led apagado")
		time.sleep(1)
		
except KeyboardInterrupt:
	print("Programa detenido")

finally:
	GPIO.cleanup()
