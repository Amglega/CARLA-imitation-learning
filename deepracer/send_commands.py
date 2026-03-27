import socket
import struct

# Using HOST and PORT from the kernel state
DEEPRACER_IP = '192.168.1.66'
PORT = 12345

def sender_script(host, port):
    # Create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            
            # Connect to the receiver
            print(f"Intentando conectar a {host}:{port}...")
            client_socket.connect((host, port))
            print(f"Conectado a {host}:{port}")
            # Two 64-bit floating-point numbers
            float1 = 3.1415926535
            float2 = 2.7182818284
            while True:
                float1 += 0.1  # Just to change the values for demonstration
                # Pack the two floats into a byte string (dd for two doubles)
                # '!' indicates network byte order (big-endian), if not specified, it's native
                # Using standard size and alignment here, for network communication, often prefer network byte order
                packed_data = struct.pack('!dd', float1, float2)

                # Send the data
                print(f"Enviando datos: float1={float1}, float2={float2}")
                client_socket.sendall(packed_data)
                print("Datos enviados correctamente.")

        except ConnectionRefusedError:
            print(f"Error: Conexión rechazada. Asegúrate de que el receptor está escuchando en {host}:{port}.")
        except Exception as e:
            print(f"Ocurrió un error: {e}")
        finally:
            # The 'with' statement handles closing the socket automatically
            print("Conexión cerrada.")


def main():
    sender_script(DEEPRACER_IP, PORT)

if __name__ == "__main__":    
    main()
