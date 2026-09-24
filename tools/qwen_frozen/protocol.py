"""Length-delimited local control messages; tensor storage stays in private tmpfs."""
import json
import struct


def send(sock, value):
    raw = json.dumps(value).encode('utf-8')
    sock.sendall(struct.pack('!I', len(raw)) + raw)


def receive(sock):
    def exact(n):
        chunks = bytearray()
        while len(chunks) < n:
            chunk = sock.recv(n - len(chunks))
            if not chunk:
                raise RuntimeError('Frozen Qwen worker connection closed')
            chunks.extend(chunk)
        return bytes(chunks)
    size = struct.unpack('!I', exact(4))[0]
    if size > 1024 * 1024:
        raise ValueError('Oversized worker control message')
    return json.loads(exact(size))
