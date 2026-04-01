import struct
import sys
import os
import numpy as np

class DNNIStructuralParser:
    def __init__(self, filename):
        self.filename = filename
        with open(filename, "rb") as f:
            self.data = f.read()
        self.offset = 0
        self.chunks = []

    def parse(self):
        print(f"--- Analyzing Structure: {os.path.basename(self.filename)} ---")

        # Global Header
        if self.data[0:4] != b'\xff\x00\xca\x7f':
            print("Error: Invalid DNNI Magic Number!")
            return

        self.offset = 8 # Skip global header

        while self.offset < len(self.data) - 8:
            marker = self.data[self.offset:self.offset+4]

            # Identify Chunk Markers: ff 40/41 ca 7f
            if marker in (b'\xff\x40\xca\x7f', b'\xff\x41\xca\x7f'):
                chunk_start = self.offset
                name_start = self.offset + 4

                # Extract Chunk Name (heuristic!!!!)
                name_bytes = self.data[name_start:name_start+12].split(b'\x00')[0]
                try:
                    name = name_bytes.decode('ascii').strip('_ \n\r')
                except:
                    name = f"chunk_{chunk_start:x}"

                # Size logic: In DNNI, size is often at chunk_start + 12 (4-byte uint32)
                size_pos = chunk_start + 12
                if size_pos + 4 <= len(self.data):
                    raw_size = struct.unpack('<I', self.data[size_pos:size_pos+4])[0]
                else:
                    raw_size = 0

                # Sanity check
                if raw_size > len(self.data) or raw_size == 0:
                    next_marker_pos = self.data.find(b'\xca\x7f', chunk_start + 4)
                    if next_marker_pos != -1:
                        estimated_size = (next_marker_pos - 2) - (chunk_start + 16)
                    else:
                        estimated_size = len(self.data) - (chunk_start + 16)

                    chunk_size = max(0, estimated_size)
                    status = "ESTIMATED"
                else:
                    chunk_size = raw_size
                    status = "HEADER-BASED"

                print(f"[{status}] Chunk: {name:<15} | At: 0x{chunk_start:08x} | Size: {chunk_size:,} bytes")

                self.chunks.append({
                    'name': name,
                    'offset': chunk_start + 16,
                    'size': chunk_size,
                    'type': 'metadata' if marker == b'\xff\x41\xca\x7f' else 'data'
                })

                # Advance to next chunk
                self.offset = chunk_start + 16 + chunk_size
            else:
                self.offset += 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <file.dnni>")
    else:
        p = DNNIStructuralParser(sys.argv[1])
        p.parse()
