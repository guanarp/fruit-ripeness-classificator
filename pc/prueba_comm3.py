import socket

def udp_server(host='0.0.0.0', port=5000):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind the socket to the host and port
    server_socket.bind((host, port))

    print(f"UDP server up and listening on {host}:{port}")

    while True:
        # Receive data from client (data, address)
        data, address = server_socket.recvfrom(4096)
        #message = data.decode('utf-8')
        print(f"Message from {address}: {data}")

if __name__ == "__main__":
    udp_server()
