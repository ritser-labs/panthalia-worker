import socket

def is_port_open(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(10)
        result = sock.connect_ex((ip, port))
        return result == 0