import socket

UDP_IP = '192.168.135.31' #Ip local
UDP_PORT = 9998

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.connect((UDP_IP,UDP_PORT))

# Continuously ask for user input to send messages
try:
    while True:
        message = input("Enter message to send (type 'exit' to quit): ")
        if message.lower() == 'exit':
            print("Exiting the sender.")
            break

        # Send message encoded as bytes
        while True:
            sock.sendto(message.encode('utf-8'), (UDP_IP, UDP_PORT))
            print(f"Message sent: {message}")

except KeyboardInterrupt:
    print("\nSender shutting down.")
finally:
    sock.close()
