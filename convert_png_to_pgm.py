#!/usr/bin/env python3
"""
Convert PNG (or PPM/PGM) stereo pairs to 8-bit grayscale P5 PGM.
Works with pure Python stdlib — no PIL/numpy required.

Usage:
  python3 convert_png_to_pgm.py left.png right.png out_left.pgm out_right.pgm

Supports:
  - 8-bit grayscale PNG  (color type 0)
  - 8-bit RGB PNG        (color type 2)  → converted to luma
  - 8-bit RGBA PNG       (color type 6)  → alpha dropped, luma
  - P5 PGM (pass-through with sanity check)
  - P6 PPM (converted to luma)
"""

import struct
import zlib
import sys
import os


# ---------------------------------------------------------------------------
# PNG decoder (stdlib only)
# ---------------------------------------------------------------------------

PNG_SIG = b'\x89PNG\r\n\x1a\n'

def _read_png(path):
    """Return (width, height, pixels_bytes_row_major_gray8)."""
    with open(path, 'rb') as f:
        data = f.read()

    if not data.startswith(PNG_SIG):
        raise ValueError(f"{path}: not a PNG file")

    pos = 8
    ihdr = None
    idat_chunks = []

    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        chunk_data = data[pos+8:pos+8+length]
        pos += 12 + length  # length + type + data + crc

        if chunk_type == b'IHDR':
            width, height = struct.unpack('>II', chunk_data[0:8])
            bit_depth   = chunk_data[8]
            color_type  = chunk_data[9]
            interlace   = chunk_data[12]
            if bit_depth != 8:
                raise ValueError(f"{path}: only 8-bit PNG supported (got {bit_depth}-bit)")
            if interlace != 0:
                raise ValueError(f"{path}: interlaced PNG not supported")
            if color_type not in (0, 2, 3, 6):
                raise ValueError(f"{path}: unsupported PNG color type {color_type}")
            ihdr = (width, height, color_type)
        elif chunk_type == b'IDAT':
            idat_chunks.append(chunk_data)
        elif chunk_type == b'IEND':
            break

    if ihdr is None:
        raise ValueError(f"{path}: missing IHDR chunk")

    width, height, color_type = ihdr
    raw = zlib.decompress(b''.join(idat_chunks))

    samples_per_pixel = {0: 1, 2: 3, 3: 1, 6: 4}[color_type]
    stride = width * samples_per_pixel

    # Undo PNG row filters
    pixels = bytearray(height * stride)
    prev_row = bytearray(stride)

    def paeth(a, b, c):
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        if pa <= pb and pa <= pc: return a
        if pb <= pc:              return b
        return c

    raw_pos = 0
    for row in range(height):
        filt = raw[raw_pos]; raw_pos += 1
        row_data = bytearray(raw[raw_pos:raw_pos + stride]); raw_pos += stride
        out = bytearray(stride)

        if filt == 0:
            out = row_data
        elif filt == 1:
            for i in range(stride):
                a = out[i - samples_per_pixel] if i >= samples_per_pixel else 0
                out[i] = (row_data[i] + a) & 0xFF
        elif filt == 2:
            for i in range(stride):
                out[i] = (row_data[i] + prev_row[i]) & 0xFF
        elif filt == 3:
            for i in range(stride):
                a = out[i - samples_per_pixel] if i >= samples_per_pixel else 0
                out[i] = (row_data[i] + (a + prev_row[i]) // 2) & 0xFF
        elif filt == 4:
            for i in range(stride):
                a = out[i - samples_per_pixel] if i >= samples_per_pixel else 0
                b = prev_row[i]
                c = prev_row[i - samples_per_pixel] if i >= samples_per_pixel else 0
                out[i] = (row_data[i] + paeth(a, b, c)) & 0xFF
        else:
            raise ValueError(f"Unknown PNG filter type {filt}")

        pixels[row * stride:(row + 1) * stride] = out
        prev_row = out

    # Convert to grayscale
    gray = bytearray(width * height)
    if color_type == 0:  # already grayscale
        gray = pixels
    elif color_type == 2:  # RGB
        for i in range(width * height):
            r, g, b = pixels[3*i], pixels[3*i+1], pixels[3*i+2]
            gray[i] = int(0.299 * r + 0.587 * g + 0.114 * b + 0.5)
    elif color_type == 6:  # RGBA
        for i in range(width * height):
            r, g, b = pixels[4*i], pixels[4*i+1], pixels[4*i+2]
            gray[i] = int(0.299 * r + 0.587 * g + 0.114 * b + 0.5)
    elif color_type == 3:  # palette (rare) — treat as gray
        gray = pixels

    return width, height, bytes(gray)


# ---------------------------------------------------------------------------
# PGM / PPM reader (for pass-through / PPM→gray conversion)
# ---------------------------------------------------------------------------

def _read_pgm_ppm(path):
    with open(path, 'rb') as f:
        data = f.read()

    def skip_whitespace_comments(pos):
        while pos < len(data):
            if data[pos:pos+1] in (b' ', b'\t', b'\n', b'\r'):
                pos += 1
            elif data[pos:pos+1] == b'#':
                while pos < len(data) and data[pos:pos+1] != b'\n':
                    pos += 1
            else:
                break
        return pos

    def read_int(pos):
        pos = skip_whitespace_comments(pos)
        start = pos
        while pos < len(data) and data[pos:pos+1].isdigit():
            pos += 1
        return int(data[start:pos]), pos

    magic = data[0:2]
    if magic not in (b'P5', b'P6'):
        raise ValueError(f"{path}: expected P5 or P6 PGM/PPM, got {magic}")

    pos = 2
    width,  pos = read_int(pos)
    height, pos = read_int(pos)
    maxval, pos = read_int(pos)
    pos += 1  # consume single whitespace after maxval

    if maxval != 255:
        raise ValueError(f"{path}: only maxval=255 supported (got {maxval})")

    if magic == b'P5':
        pixels = data[pos:]
        if len(pixels) < width * height:
            raise ValueError(f"{path}: truncated PGM data")
        return width, height, pixels[:width * height]
    else:  # P6 PPM
        rgb = data[pos:]
        if len(rgb) < width * height * 3:
            raise ValueError(f"{path}: truncated PPM data")
        gray = bytearray(width * height)
        for i in range(width * height):
            r, g, b = rgb[3*i], rgb[3*i+1], rgb[3*i+2]
            gray[i] = int(0.299 * r + 0.587 * g + 0.114 * b + 0.5)
        return width, height, bytes(gray)


# ---------------------------------------------------------------------------
# Write P5 PGM
# ---------------------------------------------------------------------------

def write_pgm(path, width, height, gray_bytes):
    with open(path, 'wb') as f:
        f.write(f"P5\n{width} {height}\n255\n".encode())
        f.write(gray_bytes)
    print(f"  Wrote {path}  ({width}x{height})")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def load_image_as_gray(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.png':
        return _read_png(path)
    elif ext in ('.pgm', '.ppm'):
        return _read_pgm_ppm(path)
    else:
        # Try PNG first, then PGM/PPM
        try:
            return _read_png(path)
        except Exception:
            return _read_pgm_ppm(path)


def convert_pair(left_in, right_in, left_out, right_out):
    print(f"Loading {left_in} ...")
    w1, h1, gray1 = load_image_as_gray(left_in)
    print(f"Loading {right_in} ...")
    w2, h2, gray2 = load_image_as_gray(right_in)

    if (w1, h1) != (w2, h2):
        raise ValueError(f"Left ({w1}x{h1}) and right ({w2}x{h2}) have different sizes")

    write_pgm(left_out, w1, h1, gray1)
    write_pgm(right_out, w2, h2, gray2)


def main():
    if len(sys.argv) == 5:
        convert_pair(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 3:
        # auto-name outputs
        base_l = os.path.splitext(os.path.basename(sys.argv[1]))[0]
        base_r = os.path.splitext(os.path.basename(sys.argv[2]))[0]
        convert_pair(sys.argv[1], sys.argv[2],
                     base_l + '.pgm', base_r + '.pgm')
    else:
        print("Usage: python3 convert_png_to_pgm.py <left> <right> [out_left.pgm out_right.pgm]")
        print("       Converts PNG/PPM stereo pairs to 8-bit grayscale P5 PGM.")
        print("       If output names are omitted, replaces the extension with .pgm")
        sys.exit(1)


if __name__ == '__main__':
    main()
