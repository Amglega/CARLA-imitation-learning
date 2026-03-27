import socket
import struct

# Using HOST and PORT from the kernel state
MASTER_IP = 'localhost'
PORT = 12345

def receiver_script(host, port):
    # Create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        try:
            
            # Bind the socket to the host and port
            server_socket.bind((host, port))
            print(f"Escuchando conexiones en {host}:{port}...")

            # Listen for incoming connections (allow up to 1 connection in queue)
            server_socket.listen(1)

            # Accept a connection
            conn, addr = server_socket.accept()
            
            with conn:
                print(f"Conectado por {addr}")
                # Receive data (expecting 16 bytes for two 64-bit floats)
                # struct.calcsize('!dd') will give 16
                data = conn.recv(struct.calcsize('!dd'))
                if data:
                    # Unpack the two floats from the received byte string
                    float1, float2 = struct.unpack('!dd', data)
                    print(f"Datos recibidos: float1={float1}, float2={float2}")
                else:
                    print("No se recibieron datos.")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            # The 'with' statement handles closing the socket automatically
            print("Conexión cerrada.")

# To run the receiver, you would call:
def main():
    receiver_script(MASTER_IP, PORT)

if __name__ == "__main__":    
    main()
