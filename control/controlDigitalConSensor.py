import RPi.GPIO as GPIO
import time

# Define pins
SENSOR_PIN = 17  # Pin for the sensor input (adjust to your setup)
OUTPUT_PIN = 18  # Pin for the digital output (adjust to your setup)
FREQ_PIN = 27

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN, GPIO.IN)  # Sensor input
GPIO.setup(OUTPUT_PIN, GPIO.OUT)  # Output control
GPIO.setup(FREQ_PIN, GPIO.OUT)  # Output control
 
# Variables
output_on_time = 6  # Output stays on for 20 seconds
last_trigger_time = 0  # To track when the sensor was last triggered

try:
    while True:
        sensor_state = GPIO.input(SENSOR_PIN)
        
        if sensor_state == GPIO.LOW:  # Sensor is ON
            print("Reading an object")
            last_trigger_time = time.time()  # Update the last trigger time
            GPIO.output(FREQ_PIN, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(OUTPUT_PIN, GPIO.HIGH)  # Turn on output
        else:
            print(f"didn't detect\t {time.time()  - last_trigger_time}")
        
        # Check if 20 seconds have passed since the last sensor trigger
        if time.time() - last_trigger_time > output_on_time:
            print("Didnt read an object in the last 20s")
            GPIO.output(OUTPUT_PIN, GPIO.LOW)  # Turn off output
            GPIO.output(FREQ_PIN, GPIO.LOW)
        
        time.sleep(0.1)  # Small delay to avoid CPU overload

except KeyboardInterrupt:
    print("Program stopped by user")
    
finally:
    GPIO.cleanup()  # Reset GPIO settings when the program is terminated

