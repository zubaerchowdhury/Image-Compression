#!/usr/bin/env python3
"""
img-comp-algo.py -- a JPEG-like image codec written from first principles.

Nothing in the compression path calls an image-codec library. The pipeline is:

    [Input RGB image]
          |
          v
    [Colour conversion]     RGB -> YCbCr, separating brightness from colour
          |
          v
    [Chroma subsampling]    the two colour planes are halved in each direction (4:2:0)
          |
          v
    [Block splitting]       every plane is cut into 8x8 pixel blocks
          |
          v
    [Frequency transform]   a 2-D DCT turns each block into 64 frequency coefficients
          |
          v
    [Quantisation]          coefficients are divided by a step size and rounded (lossy)
          |
          v
    [Entropy coding]        zig-zag scan, run-length tokens, Huffman codes, packed bits

The packed bits are written as a baseline JPEG stream and stored in a container
that keeps the input file's extension:

    .jpg / .jpeg   the JPEG stream itself (a normal JFIF file)
    .tif / .tiff   a TIFF whose single strip holds the stream (Compression = 7, "JPEG")
    .bmp           a BMP whose pixel block holds the stream (biCompression = 4, "BI_JPEG")

A second mode, "8bit-quant", re-quantises 16-bit TIFFs to 8-bit TIFFs.

Terminal use
------------
    python img-comp-algo.py                               use the settings below
    python img-comp-algo.py Assets/alo_UV.bmp             compress one file
    python img-comp-algo.py Assets --quality 30           compress every image in a folder
    python img-comp-algo.py Assets/x.tif --mode 8bit-quant [--stretch]
    python img-comp-algo.py --help                        all options
"""

# %% Imports
import argparse
import array
import heapq
import struct
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# %% Settings
# One path string works on Windows, macOS and Linux: use forward slashes and keep
# it relative to this file, so the project can live in any folder on any machine.
IMAGE_PATH = "Assets/alo_visible.bmp"
MODE = "compress"            # "compress"  or  "8bit-quant"
QUALITY = 50                 # 1 (tiny file, blocky) ... 100 (large file, near-lossless)
OUTPUT_DIR = "output"        # results are written here, next to this file
STRETCH_16_TO_8 = False      # 8bit-quant: True stretches the 0.1..99.9 percentile range to 0..255
VERIFY = True                # compress: decode the result again, report PSNR, write a PNG preview

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SUPPORTED = {".tif": "tif", ".tiff": "tif", ".jpg": "jpg", ".jpeg": "jpg", ".bmp": "bmp"}

# %% Loading images
# resolves a path string (forward or back slashes) relative to this file's folder
def resolve_path(path_str):
    path = Path(str(path_str).replace("\\", "/")).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


# reads a TIFF with tifffile and reduces it to one page shaped (H, W) or (H, W, 3)
def load_tiff_array(path):
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (2, 3, 4):
        arr = arr.transpose(1, 2, 0)      # planar (channels first) -> channels last
    while arr.ndim > 3 or (arr.ndim == 3 and arr.shape[-1] not in (2, 3, 4)):
        arr = arr[0]                      # multi-page or stacked file: keep the first page
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]                # drop an alpha channel
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[..., 0]                 # grey + alpha -> grey
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise ValueError(f"unsupported TIFF sample type {arr.dtype}")
    return arr


# loads a .tif/.jpg/.bmp file as a numpy array shaped (H, W) grey or (H, W, 3) RGB
def load_image(path):
    path = resolve_path(path)
    kind = SUPPORTED.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"{path.name}: only {', '.join(SUPPORTED)} files are supported")
    if kind == "tif":
        stream = tiff_extract_jpeg(path)                 # a file this program wrote earlier?
        return decompress_image(stream) if stream is not None else load_tiff_array(path)
    if kind == "bmp":
        stream = bmp_extract_jpeg(path)
        if stream is not None:
            return decompress_image(stream)
    img = Image.open(path)
    if img.mode not in ("L", "RGB"):
        img = img.convert("L" if img.mode in ("1", "LA", "I", "I;16", "F") else "RGB")
    return np.asarray(img)

# %% 16-bit to 8-bit requantisation
# re-quantises a 16-bit (or float) array to 8-bit, over the full nominal range or stretched to the data range
def quantize_16_to_8(arr, stretch=False):
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    data = arr.astype(np.float32)
    if stretch:
        sample = data.reshape(-1)[::max(1, data.size // 2_000_000)]   # enough samples for stable percentiles
        lo, hi = (float(x) for x in np.percentile(sample, (0.1, 99.9)))  # ignore a few outliers at both ends
    elif np.issubdtype(arr.dtype, np.integer):
        lo, hi = 0.0, float(np.iinfo(arr.dtype).max)  # 65535 for uint16: 257 input levels per output level
    else:                                             # floats: 0..1 unless values clearly exceed it (allow overshoot)
        top = float(np.nanmax(data))
        lo, hi = 0.0, 1.0 if top <= 2.0 else 255.0 if top <= 300.0 else 65535.0 if top <= 70000.0 else top
    if hi <= lo:
        hi = lo + 1.0
    scaled = (data - lo) * (255.0 / (hi - lo))
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)

# %% Colour conversion
KR, KB = 0.299, 0.114          # share of perceived brightness that comes from red and blue
KG = 1.0 - KR - KB             # green supplies the rest (0.587)


# converts an RGB uint8 image to Y (brightness), Cb and Cr (colour differences) float planes
def rgb_to_ycbcr(rgb):
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    y = KR * r + KG * g + KB * b                 # weighted brightness
    cb = (b - y) / (2.0 * (1.0 - KB)) + 128.0    # "how much bluer than grey", centred on 128
    cr = (r - y) / (2.0 * (1.0 - KR)) + 128.0    # "how much redder than grey", centred on 128
    return y, cb, cr


# converts Y, Cb, Cr float planes back to an RGB uint8 image
def ycbcr_to_rgb(y, cb, cr):
    cb = cb - 128.0
    cr = cr - 128.0
    r = y + 2.0 * (1.0 - KR) * cr
    b = y + 2.0 * (1.0 - KB) * cb
    g = (y - KR * r - KB * b) / KG               # the brightness equation solved for green
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

# %% Chroma subsampling and block splitting
# pads a 2-D plane by repeating its edge pixels until both sides are multiples of `multiple`
def pad_plane(plane, multiple):
    h, w = plane.shape
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    if pad_h or pad_w:
        plane = np.pad(plane, ((0, pad_h), (0, pad_w)), mode="edge")
    return plane


# halves a plane in both directions (4:2:0) by averaging every 2x2 group of samples
def chroma_subsample(plane):
    h, w = plane.shape
    return plane.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


# doubles a plane in both directions with a triangle filter: each new sample is 3/4 its nearest old sample + 1/4 the next
def chroma_upsample(plane):
    for axis in (0, 1):
        n = plane.shape[axis]
        before = np.take(plane, np.maximum(np.arange(n) - 1, 0), axis=axis)      # low-side neighbour (edge repeated)
        after = np.take(plane, np.minimum(np.arange(n) + 1, n - 1), axis=axis)   # high-side neighbour
        low = (3.0 * plane + before) / 4.0
        high = (3.0 * plane + after) / 4.0
        shape = plane.shape[:axis] + (2 * n,) + plane.shape[axis + 1:]
        plane = np.stack([low, high], axis=axis + 1).reshape(shape)             # interleave low, high, low, high ...
    return plane


# cuts a plane whose sides are multiples of 8 into 8x8 blocks shaped (rows, cols, 8, 8)
def split_blocks(plane):
    h, w = plane.shape
    return plane.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)


# joins (rows, cols, 8, 8) blocks back into one plane
def merge_blocks(blocks):
    rows, cols = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(rows * 8, cols * 8)


# orders a component's blocks as they appear in the scan: MCU by MCU, hv x hv blocks row-major inside each MCU
def blocks_in_scan_order(blocks, hv):
    rows, cols = blocks.shape[:2]
    grouped = blocks.reshape(rows // hv, hv, cols // hv, hv, 8, 8).transpose(0, 2, 1, 3, 4, 5)
    return grouped.reshape(-1, 8, 8)


# undoes blocks_in_scan_order: (N, 8, 8) scan-ordered blocks back to a (rows, cols, 8, 8) grid
def blocks_from_scan_order(blocks, hv, mcu_rows, mcu_cols):
    grouped = blocks.reshape(mcu_rows, mcu_cols, hv, hv, 8, 8).transpose(0, 2, 1, 3, 4, 5)
    return grouped.reshape(mcu_rows * hv, mcu_cols * hv, 8, 8)

# %% Discrete cosine transform
# builds the 8x8 DCT-II basis matrix T[u, x] = c(u) * cos((2x + 1) * u * pi / 16)
def dct_matrix():
    x = np.arange(8)
    u = x.reshape(8, 1)
    basis = np.cos((2 * x + 1) * u * np.pi / 16) * np.sqrt(2.0 / 8.0)
    basis[0, :] = np.sqrt(1.0 / 8.0)             # the constant (DC) row is scaled so T is orthonormal
    return basis.astype(np.float32)


DCT_T = dct_matrix()


# applies the forward 2-D DCT to every block (level-shifted by -128): F = T . B . T^T
def dct_blocks(blocks):
    return DCT_T @ (blocks.astype(np.float32) - 128.0) @ DCT_T.T


# applies the inverse 2-D DCT to every coefficient block and undoes the level shift: B = T^T . F . T + 128
def idct_blocks(coeffs):
    return DCT_T.T @ coeffs.astype(np.float32) @ DCT_T + 128.0

# %% Quantisation
# builds the luma and chroma quantisation tables for a quality of 1..100 (step size grows with frequency)
def quant_tables(quality):
    quality = int(np.clip(quality, 1, 100))
    scale = 50.0 / quality if quality < 50 else (100 - quality) / 50.0
    u = np.arange(8).reshape(8, 1)
    v = np.arange(8)
    distance = u + v                      # 0 at the DC corner, 14 at the highest frequency
    luma = 8 + 6 * distance               # 8 .. 92   brightness detail is what the eye notices most
    chroma = 16 + 12 * distance           # 16 .. 184 colour detail can be thrown away more freely
    tables = []
    for base in (luma, chroma):
        table = np.floor(base * scale + 0.5)
        tables.append(np.clip(table, 1, 255).astype(np.int32))
    return tables[0], tables[1]


# divides every DCT coefficient by its step size and rounds to an integer (the lossy step)
def quantize(coeffs, table):
    return np.round(coeffs / table.astype(np.float32)).astype(np.int32)


# multiplies quantised coefficients back by their step sizes
def dequantize(qcoeffs, table):
    return qcoeffs.astype(np.float32) * table.astype(np.float32)

# %% Zig-zag scan and run-length tokens
# builds the zig-zag order: walks the 8x8 grid along anti-diagonals so low frequencies come first
def zigzag_order():
    order = []
    for s in range(15):                               # s = row + col, the anti-diagonal index
        rows = range(max(0, s - 7), min(s, 7) + 1)
        if s % 2 == 0:
            rows = reversed(rows)                     # even diagonals run upwards, odd ones downwards
        for r in rows:
            order.append(r * 8 + (s - r))
    return np.array(order)


ZIGZAG = zigzag_order()                 # position i of the scan -> flat index into the 8x8 block
UNZIGZAG = np.argsort(ZIGZAG)           # flat index into the 8x8 block -> position in the scan


# flattens (N, 8, 8) blocks into (N, 64) vectors in zig-zag order
def zigzag(blocks):
    return blocks.reshape(-1, 64)[:, ZIGZAG]


# restores (N, 64) zig-zag vectors into (N, 8, 8) blocks
def unzigzag(vectors):
    return vectors[:, UNZIGZAG].reshape(-1, 8, 8)


# returns how many bits are needed to hold |v| (JPEG's "size category"), 0 for v == 0
def bit_size(values):
    magnitude = np.abs(np.asarray(values, dtype=np.int64)).astype(np.float64)
    return np.frexp(magnitude)[1].astype(np.uint8)    # exponent e with 2^(e-1) <= |v| < 2^e


# returns the extra bits that follow a size category: v itself if positive, else v - 1 in `size` bits
def value_bits(values, size):
    values = np.asarray(values, dtype=np.int64)
    negative = values + (np.int64(1) << size.astype(np.int64)) - 1
    return np.where(values >= 0, values, negative).astype(np.uint32)


# gives every token a sort key so tokens from all components can be put into scan order
def token_key(block, position, sub):
    return (np.asarray(block, dtype=np.int64) * 65 + position) * 4 + sub


# turns one component's (N, 64) zig-zag blocks into DC/AC tokens; `scan_pos` is each block's place in the scan
def tokenize(zz, scan_pos):
    n_blocks = zz.shape[0]
    u8 = np.uint8
    # DC: code the difference from the previous block's DC of the same component
    diff = np.diff(zz[:, 0].astype(np.int64), prepend=0)
    dc_size = bit_size(diff)
    dc = dict(sym=dc_size, extra=value_bits(diff, dc_size), nbits=dc_size,
              kind=np.zeros(n_blocks, u8), key=token_key(scan_pos, 0, 0))
    # AC: every non-zero coefficient becomes the symbol (zeros before it, size) plus its value bits
    rows, cols = np.nonzero(zz[:, 1:])
    cols = cols + 1
    values = zz[rows, cols]
    prev_cols = np.concatenate(([0], cols[:-1]))
    same_block = np.concatenate(([False], rows[1:] == rows[:-1]))
    run = cols - np.where(same_block, prev_cols, 0) - 1
    zrl_count = run // 16                                  # runs of 16+ zeros need ZRL tokens first
    run = run % 16
    ac_size = bit_size(values)
    ac = dict(sym=((run << 4) | ac_size).astype(u8), extra=value_bits(values, ac_size), nbits=ac_size,
              kind=np.ones(len(rows), u8), key=token_key(scan_pos[rows], cols, 3))
    # ZRL: the symbol 0xF0 stands for sixteen zeros; repeated zrl_count times before the coefficient
    zrl_rows = np.repeat(rows, zrl_count)
    zrl_cols = np.repeat(cols, zrl_count)
    first = np.cumsum(zrl_count) - zrl_count
    zrl_sub = np.arange(len(zrl_rows)) - np.repeat(first, zrl_count)
    zrl = dict(sym=np.full(len(zrl_rows), 0xF0, u8), extra=np.zeros(len(zrl_rows), np.uint32),
               nbits=np.zeros(len(zrl_rows), u8), kind=np.ones(len(zrl_rows), u8),
               key=token_key(scan_pos[zrl_rows], zrl_cols, zrl_sub))
    # EOB: the symbol 0x00 ends a block early, unless its very last coefficient is non-zero
    last = np.zeros(n_blocks, dtype=np.int64)
    np.maximum.at(last, rows, cols)
    eob_blocks = np.flatnonzero(last < 63)
    eob = dict(sym=np.zeros(len(eob_blocks), u8), extra=np.zeros(len(eob_blocks), np.uint32),
               nbits=np.zeros(len(eob_blocks), u8), kind=np.ones(len(eob_blocks), u8),
               key=token_key(scan_pos[eob_blocks], 64, 0))
    parts = (dc, zrl, ac, eob)
    return {field: np.concatenate([p[field] for p in parts]) for field in dc}

# %% Huffman coding
# builds Huffman code lengths (max 16 bits, as JPEG requires) from symbol counts; returns BITS and ordered symbols
def huffman_code_lengths(counts):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() == 0:
        return np.zeros(16, dtype=np.int64), []
    # Each heap node is (weight, tie-breaker, symbols below it). A dummy symbol 256 of weight 1
    # takes the longest code, so no real symbol ends up with an all-ones code (JPEG forbids those).
    lengths = np.zeros(257, dtype=np.int64)
    heap = [(int(c), s, [s]) for s, c in enumerate(counts) if c > 0] + [(1, 256, [256])]
    heapq.heapify(heap)
    while len(heap) > 1:
        w1, t1, s1 = heapq.heappop(heap)
        w2, t2, s2 = heapq.heappop(heap)
        lengths[s1] += 1                   # everything under the merged node moves one level deeper
        lengths[s2] += 1
        heapq.heappush(heap, (w1 + w2, min(t1, t2), s1 + s2))
    # JPEG allows at most 16 bits per code: move pairs of over-long codes up (the Kraft sum is unchanged)
    bits = np.bincount(lengths[lengths > 0], minlength=34)
    for length in range(len(bits) - 1, 16, -1):
        while bits[length] > 0:
            j = length - 2
            while bits[j] == 0:
                j -= 1
            bits[length] -= 2
            bits[length - 1] += 1
            bits[j + 1] += 2
            bits[j] -= 1
    longest = max(i for i in range(1, 17) if bits[i] > 0)
    bits[longest] -= 1                     # drop the dummy symbol, which owns the longest code
    # canonical assignment: the most frequent symbols get the shortest codes
    symbols = [int(s) for s in np.argsort(-counts, kind="stable") if counts[s] > 0]
    return bits[1:17], symbols


# derives the canonical codes (code, length) for every symbol from the BITS table and the ordered symbols
def huffman_codes(bits, symbols):
    codes = np.zeros(256, dtype=np.uint32)
    lengths = np.zeros(256, dtype=np.uint8)
    code, i = 0, 0
    for length in range(1, 17):
        for _ in range(int(bits[length - 1])):
            codes[symbols[i]] = code       # codes of one length are consecutive integers ...
            lengths[symbols[i]] = length
            code += 1
            i += 1
        code <<= 1                         # ... and the next length starts one bit longer
    return codes, lengths


# builds 65536-entry look-up tables (symbol, code length) so a 16-bit peek decodes one Huffman symbol
def huffman_lut(bits, symbols):
    sym = np.zeros(65536, dtype=np.uint8)
    length = np.zeros(65536, dtype=np.uint8)
    code, i = 0, 0
    for size in range(1, 17):
        for _ in range(int(bits[size - 1])):
            start, end = code << (16 - size), (code + 1) << (16 - size)
            sym[start:end] = symbols[i]    # every 16-bit window that begins with this code
            length[start:end] = size
            code += 1
            i += 1
        code <<= 1
    return bytes(sym), bytes(length)

# %% Bit packing
# packs variable-length fields (values[i] in lengths[i] bits) into bytes, most significant bit first, 1-padded
def pack_bits(values, lengths, chunk=200_000):
    pieces = []
    for start in range(0, len(values), chunk):
        v = values[start:start + chunk].astype(np.uint32)
        n = lengths[start:start + chunk].astype(np.int64)
        total = int(n.sum())
        offsets = np.cumsum(n) - n                                   # first bit of each field
        bit_in_field = np.arange(total) - np.repeat(offsets, n)      # 0, 1, 2 ... inside a field
        shift = (np.repeat(n, n) - 1 - bit_in_field).astype(np.uint32)
        pieces.append(((np.repeat(v, n) >> shift) & 1).astype(np.uint8))
    bits = np.concatenate(pieces) if pieces else np.zeros(0, np.uint8)
    pad = (-len(bits)) % 8
    bits = np.concatenate([bits, np.ones(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


# inserts a 0x00 after every 0xFF data byte so a decoder never mistakes data for a marker
def stuff_bytes(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = 1 + (arr == 0xFF).astype(np.int64)
    out = np.zeros(int(counts.sum()), dtype=np.uint8)
    out[np.cumsum(counts) - counts] = arr
    return out.tobytes()


# removes the stuffed 0x00 bytes that follow 0xFF in entropy-coded data
def unstuff_bytes(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    drop = np.zeros(len(arr), dtype=bool)
    drop[1:] = (arr[:-1] == 0xFF) & (arr[1:] == 0x00)
    return arr[~drop].tobytes()


# precomputes the 16 bits that start at every bit position, so one look-up gives a whole Huffman code
def bit_windows(raw):
    n = len(raw)
    b = np.frombuffer(raw + b"\0\0\0", dtype=np.uint8).astype(np.uint32)
    w24 = (b[:n] << 16) | (b[1:n + 1] << 8) | b[2:n + 2]        # 24 bits starting at each byte
    windows = np.empty((n, 8), dtype=np.uint32)
    for s in range(8):                                           # bit offset inside the byte
        windows[:, s] = (w24 >> (8 - s)) & 0xFFFF
    return array.array("I", windows.reshape(-1).tobytes())

# %% JPEG stream assembly
# writes one marker segment: 0xFF, the marker code, a 2-byte length that counts itself, then the payload
def segment(marker, payload):
    return b"\xFF" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


# assembles a complete baseline JFIF stream from the image size, components, tables and scan data
def build_jpeg(width, height, components, qtables, htables, scan_data):
    out = [b"\xFF\xD8"]                                                        # SOI
    out.append(segment(0xE0, b"JFIF\0" + bytes([1, 1, 0]) + struct.pack(">HH", 1, 1) + b"\0\0"))
    for tq, table in qtables:                                                  # DQT, values in zig-zag order
        out.append(segment(0xDB, bytes([tq]) + table.reshape(64)[ZIGZAG].astype(np.uint8).tobytes()))
    sof = struct.pack(">BHHB", 8, height, width, len(components))              # SOF0: 8-bit baseline
    for cid, h, v, tq, _, _ in components:
        sof += bytes([cid, (h << 4) | v, tq])
    out.append(segment(0xC0, sof))
    for tc, th, bits, symbols in htables:                                      # DHT
        out.append(segment(0xC4, bytes([(tc << 4) | th]) + bytes(int(b) for b in bits) + bytes(symbols)))
    sos = bytes([len(components)])                                             # SOS
    for cid, _, _, _, td, ta in components:
        sos += bytes([cid, (td << 4) | ta])
    out.append(segment(0xDA, sos + bytes([0, 63, 0])))
    out.append(scan_data)
    out.append(b"\xFF\xD9")                                                    # EOI
    return b"".join(out)

# %% TIFF and BMP containers
# wraps a JPEG stream in a .tif container as one JPEG-compressed strip (TIFF Compression = 7)
def wrap_tiff(stream, width, height, color):
    SHORT, LONG, RATIONAL = 3, 4, 5
    spp = 3 if color else 1
    entries = [
        (256, LONG, [width]), (257, LONG, [height]), (258, SHORT, [8] * spp),
        (259, SHORT, [7]), (262, SHORT, [6 if color else 1]),               # JPEG, YCbCr or greyscale
        (273, LONG, None),                                                  # StripOffsets, filled below
        (277, SHORT, [spp]), (278, LONG, [height]), (279, LONG, [len(stream)]),
        (282, RATIONAL, [(72, 1)]), (283, RATIONAL, [(72, 1)]),
        (284, SHORT, [1]), (296, SHORT, [2]),
    ]
    if color:
        entries += [(530, SHORT, [2, 2]), (531, SHORT, [1])]                # 4:2:0 chroma, centred
    ifd_offset = 8
    extra_offset = ifd_offset + 2 + 12 * len(entries) + 4
    extra = b""                                                             # values too big for 4 bytes go after the IFD
    packed = []
    data_offset = extra_offset
    for tag, typ, values in entries:
        if typ == RATIONAL:
            raw = b"".join(struct.pack("<II", n, d) for n, d in values)
        elif values is None:
            raw = b""
        else:
            raw = struct.pack("<" + ("H" if typ == SHORT else "I") * len(values), *values)
        count = len(values) if values is not None else 1
        if values is not None and len(raw) > 4:
            packed.append((tag, typ, count, struct.pack("<I", data_offset)))
            extra += raw + (b"\0" if len(raw) % 2 else b"")
            data_offset = extra_offset + len(extra)
        else:
            packed.append((tag, typ, count, raw.ljust(4, b"\0")))
    strip_offset = data_offset + (-data_offset) % 4
    extra += b"\0" * (strip_offset - data_offset)
    ifd = struct.pack("<H", len(packed))
    for tag, typ, count, value in packed:
        if tag == 273:
            value = struct.pack("<I", strip_offset)
        ifd += struct.pack("<HHI", tag, typ, count) + value
    ifd += struct.pack("<I", 0)                                             # no further IFDs
    return b"II" + struct.pack("<HI", 42, ifd_offset) + ifd + extra + stream


# returns the JPEG stream stored in a single-strip JPEG-compressed .tif file, or None for any other TIFF
def tiff_extract_jpeg(path):
    with open(path, "rb") as f:
        head = f.read(8)
        if head[:2] not in (b"II", b"MM"):
            return None
        e = "<" if head[:2] == b"II" else ">"                               # byte order: little or big endian
        if struct.unpack_from(e + "H", head, 2)[0] != 42:
            return None                                                     # BigTIFF etc.: leave it to tifffile
        f.seek(struct.unpack_from(e + "I", head, 4)[0])
        n = struct.unpack(e + "H", f.read(2))[0]
        raw = f.read(12 * n)
        tags = {}
        for i in range(n):
            tag, typ, count = struct.unpack_from(e + "HHI", raw, 12 * i)
            if count == 1 and typ in (3, 4):
                tags[tag] = struct.unpack_from(e + ("H" if typ == 3 else "I"), raw, 12 * i + 8)[0]
            else:
                tags[tag] = None                                            # multi-valued: not a single strip
        if tags.get(259) != 7 or tags.get(273) is None or tags.get(279) is None or 347 in tags:
            return None
        f.seek(tags[273])
        return f.read(tags[279])


# wraps a JPEG stream in a .bmp container using the BI_JPEG compression type of the BMP specification
def wrap_bmp(stream, width, height):
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 0, 4, len(stream), 2835, 2835, 0, 0)
    header = struct.pack("<2sIHHI", b"BM", 14 + 40 + len(stream), 0, 0, 14 + 40)
    return header + info + stream


# returns the JPEG stream stored inside a BI_JPEG .bmp file, or None if the file is an ordinary bitmap
def bmp_extract_jpeg(path):
    with open(path, "rb") as f:
        head = f.read(54)
        if head[:2] != b"BM" or len(head) < 54 or struct.unpack_from("<I", head, 30)[0] != 4:
            return None
        offset = struct.unpack_from("<I", head, 10)[0]
        size = struct.unpack_from("<I", head, 34)[0]
        f.seek(offset)
        return f.read(size) if size else f.read()

# %% The encoder
# runs the whole pipeline on an 8-bit image array and returns a complete baseline JPEG byte stream
def compress_image(image, quality=50):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = quantize_16_to_8(image)
    height, width = image.shape[:2]
    color = image.ndim == 3
    luma_q, chroma_q = quant_tables(quality)
    if color:
        y, cb, cr = rgb_to_ycbcr(image)
        planes = [(pad_plane(y, 16), 0, 2),                                   # (plane, table id, sampling)
                  (chroma_subsample(pad_plane(cb, 16)), 1, 1),
                  (chroma_subsample(pad_plane(cr, 16)), 1, 1)]
        components = [(1, 2, 2, 0, 0, 0), (2, 1, 1, 1, 1, 1), (3, 1, 1, 1, 1, 1)]   # (id, h, v, tq, td, ta)
    else:
        planes = [(pad_plane(image.astype(np.float32), 8), 0, 1)]
        components = [(1, 1, 1, 0, 0, 0)]
    tables = (luma_q, chroma_q)
    blocks_per_mcu = [hv * hv for _, _, hv in planes]
    offsets = np.cumsum([0] + blocks_per_mcu[:-1])
    mcu_blocks = sum(blocks_per_mcu)
    # stages 3-6 per component: blocks -> DCT -> quantise -> zig-zag -> tokens
    parts = []
    for c, (plane, table_id, hv) in enumerate(planes):
        blocks = blocks_in_scan_order(split_blocks(plane), hv)
        zz = zigzag(quantize(dct_blocks(blocks), tables[table_id]))
        index = np.arange(zz.shape[0], dtype=np.int64)
        scan_pos = mcu_blocks * (index // blocks_per_mcu[c]) + offsets[c] + index % blocks_per_mcu[c]
        tokens = tokenize(zz, scan_pos)
        tokens["table"] = np.full(len(tokens["sym"]), table_id, dtype=np.uint8)
        parts.append(tokens)
    tokens = {field: np.concatenate([p[field] for p in parts]) for field in parts[0]}
    order = np.argsort(tokens["key"], kind="stable")
    tokens = {field: values[order] for field, values in tokens.items()}
    # stage 7: Huffman tables from the symbol statistics (DC/AC x luma/chroma), then the bit stream
    htables = []
    codes = np.zeros((2, 2, 256), dtype=np.uint32)
    code_lengths = np.zeros((2, 2, 256), dtype=np.uint8)
    for kind in (0, 1):
        for table_id in range(2 if color else 1):
            mask = (tokens["kind"] == kind) & (tokens["table"] == table_id)
            bits, symbols = huffman_code_lengths(np.bincount(tokens["sym"][mask], minlength=256))
            codes[kind, table_id], code_lengths[kind, table_id] = huffman_codes(bits, symbols)
            htables.append((kind, table_id, bits, symbols))
    code = codes[tokens["kind"], tokens["table"], tokens["sym"]]
    clen = code_lengths[tokens["kind"], tokens["table"], tokens["sym"]]
    values = (code << tokens["nbits"].astype(np.uint32)) | tokens["extra"]   # Huffman code then value bits
    lengths = clen.astype(np.int64) + tokens["nbits"]
    scan = stuff_bytes(pack_bits(values, lengths))
    qtables = [(0, luma_q), (1, chroma_q)] if color else [(0, luma_q)]
    return build_jpeg(width, height, components, qtables, htables, scan)

# %% The decoder
# decodes the entropy-coded scan into (N, 64) quantised zig-zag blocks per component, in raster block order
def decode_scan(scan, comps, luts, width, height):
    window = bit_windows(unstuff_bytes(scan))
    hmax = max(h for _, h, _, _ in comps)
    vmax = max(v for _, _, v, _ in comps)
    mcus_x = -(-width // (8 * hmax))
    mcus_y = -(-height // (8 * vmax))
    coef, plan = [], []
    for ci, (cid, h, v, tq) in enumerate(comps):
        cols = mcus_x * h
        coef.append(array.array("i", bytes(4 * cols * mcus_y * v * 64)))
        dc_sym, dc_len, ac_sym, ac_len = luts[cid]
        for j in range(v):
            for i in range(h):
                plan.append((ci, h, v, cols, i, j, dc_sym, dc_len, ac_sym, ac_len))
    pred = [0] * len(comps)
    pos = 0
    for my in range(mcus_y):
        for mx in range(mcus_x):
            for ci, h, v, cols, i, j, dc_sym, dc_len, ac_sym, ac_len in plan:
                out = coef[ci]
                base = ((my * v + j) * cols + mx * h + i) * 64
                w = window[pos]                                  # DC: size category, then value bits
                s, n = dc_sym[w], dc_len[w]
                if n == 0:
                    raise ValueError("corrupt Huffman data")
                pos += n
                if s:
                    bits = window[pos] >> (16 - s)
                    pos += s
                    if bits < (1 << (s - 1)):
                        bits -= (1 << s) - 1                     # negative values were stored as v - 1
                    pred[ci] += bits
                out[base] = pred[ci]
                k = 1
                while k < 64:                                    # AC: (run, size) symbols until EOB
                    w = window[pos]
                    s, n = ac_sym[w], ac_len[w]
                    if n == 0:
                        raise ValueError("corrupt Huffman data")
                    pos += n
                    if s == 0:
                        break                                    # EOB: the rest of the block is zero
                    run, size = s >> 4, s & 15
                    if size == 0:
                        k += 16                                  # ZRL: sixteen zeros
                        continue
                    k += run
                    bits = window[pos] >> (16 - size)
                    pos += size
                    if bits < (1 << (size - 1)):
                        bits -= (1 << size) - 1
                    out[base + k] = bits
                    k += 1
    blocks = [np.frombuffer(c, dtype=np.int32).reshape(-1, 64) for c in coef]
    return blocks, mcus_x, mcus_y


# parses a baseline JPEG stream and rebuilds the image: Huffman decode -> dequantise -> IDCT -> upsample -> RGB
def decompress_image(stream):
    data = bytes(stream)
    if data[:2] != b"\xFF\xD8":
        raise ValueError("not a JPEG stream")
    qtables, luts, comps = {}, {}, []
    width = height = 0
    pos = 2
    while pos < len(data):
        if data[pos] != 0xFF:
            raise ValueError(f"expected a marker at byte {pos}")
        marker = data[pos + 1]
        pos += 2
        if marker in (0xFF, 0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            continue                                             # fill bytes and stand-alone markers
        if marker == 0xD9:
            break                                                # EOI
        length = struct.unpack_from(">H", data, pos)[0]
        payload = data[pos + 2:pos + length]
        pos += length
        if marker == 0xDB:                                       # DQT: one or more tables
            p = 0
            while p < len(payload):
                precision, tq = payload[p] >> 4, payload[p] & 15
                size = 2 if precision else 1
                raw = payload[p + 1:p + 1 + 64 * size]
                values = np.frombuffer(raw, dtype=">u2" if precision else np.uint8).astype(np.int32)
                qtables[tq] = values[UNZIGZAG].reshape(8, 8)
                p += 1 + 64 * size
        elif marker == 0xC0:                                     # SOF0: baseline frame header
            precision, height, width, n = struct.unpack_from(">BHHB", payload, 0)
            if precision != 8:
                raise ValueError("only 8-bit samples are supported")
            comps = [(payload[6 + 3 * i], payload[7 + 3 * i] >> 4, payload[7 + 3 * i] & 15, payload[8 + 3 * i])
                     for i in range(n)]
        elif marker in (0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            raise ValueError("only baseline (sequential, Huffman) JPEG streams are supported")
        elif marker == 0xC4:                                     # DHT: one or more Huffman tables
            p = 0
            while p < len(payload):
                tc, th = payload[p] >> 4, payload[p] & 15
                bits = list(payload[p + 1:p + 17])
                symbols = list(payload[p + 17:p + 17 + sum(bits)])
                luts[(tc, th)] = huffman_lut(bits, symbols)
                p += 17 + sum(bits)
        elif marker == 0xDD:
            raise ValueError("restart intervals are not supported")
        elif marker == 0xDA:                                     # SOS: scan header, then the coded data
            n = payload[0]
            comp_luts = {}
            for i in range(n):
                cid, tables = payload[1 + 2 * i], payload[2 + 2 * i]
                comp_luts[cid] = luts[(0, tables >> 4)] + luts[(1, tables & 15)]
            end = pos
            while True:                                          # the data ends at the first real marker
                end = data.find(b"\xFF", end)
                if end < 0:
                    end = len(data)
                    break
                nxt = data[end + 1] if end + 1 < len(data) else 0xD9
                if nxt != 0 and not (0xD0 <= nxt <= 0xD7):
                    break
                end += 2
            blocks, mcus_x, mcus_y = decode_scan(data[pos:end], comps, comp_luts, width, height)
            pos = end
    # stages 6-1 in reverse for every component
    hmax = max(h for _, h, _, _ in comps)
    vmax = max(v for _, _, v, _ in comps)
    planes = []
    for ci, (cid, h, v, tq) in enumerate(comps):
        pixels = idct_blocks(dequantize(unzigzag(blocks[ci]), qtables[tq]))
        plane = merge_blocks(pixels.reshape(mcus_y * v, mcus_x * h, 8, 8))
        if (vmax // v, hmax // h) == (2, 2):
            plane = chroma_upsample(plane)                       # 4:2:0 chroma back to full size
        else:
            plane = plane.repeat(vmax // v, axis=0).repeat(hmax // h, axis=1)
        planes.append(plane[:height, :width])
    if len(planes) == 3:
        return ycbcr_to_rgb(*planes)
    return np.clip(np.round(planes[0]), 0, 255).astype(np.uint8)

# %% Quality metric
# computes the peak signal-to-noise ratio in dB between two 8-bit images of the same shape
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


# formats a byte count with thousands separators
def fmt_bytes(n):
    return f"{n:,} bytes"

# %% Running the two modes
# compresses one file, writes the smaller same-format file (+ a lossless PNG preview) and prints a report
def run_compress(path, quality=QUALITY, out_dir=OUTPUT_DIR, verify=VERIFY):
    path = resolve_path(path)
    out_dir = resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = SUPPORTED[path.suffix.lower()]
    print(f"[compress] {path}")
    started = time.perf_counter()
    image = load_image(path)
    if image.dtype != np.uint8:
        print(f"  {image.dtype} samples -> re-quantised to 8-bit before compression")
        image = quantize_16_to_8(image)
    height, width = image.shape[:2]
    color = image.ndim == 3
    stream = compress_image(image, quality)
    if kind == "jpg":
        payload, note = stream, "baseline JFIF JPEG"
    elif kind == "tif":
        payload, note = wrap_tiff(stream, width, height, color), "TIFF, Compression=7 (JPEG) strip"
    else:
        payload, note = wrap_bmp(stream, width, height), "BMP, BI_JPEG pixel block"
    out_path = out_dir / f"{path.stem}_q{quality}{path.suffix}"
    out_path.write_bytes(payload)
    elapsed = time.perf_counter() - started
    in_size = path.stat().st_size
    raw_size = image.size
    print(f"  input : {width} x {height} {'RGB' if color else 'grey'}, {fmt_bytes(in_size)} on disk"
          f" ({fmt_bytes(raw_size)} as raw 8-bit pixels)")
    print(f"  output: {out_path.relative_to(BASE_DIR) if out_path.is_relative_to(BASE_DIR) else out_path}"
          f"  [{note}]  {fmt_bytes(len(payload))}")
    print(f"          {in_size / len(payload):.1f}x smaller than the input file,"
          f" {8 * len(payload) / raw_size:.2f} bits per pixel, quality {quality}, {elapsed:.1f} s")
    if kind == "bmp":
        print("          note: BI_JPEG bitmaps open in this program and a few viewers only; see the PNG preview")
    if verify:
        started = time.perf_counter()
        decoded = decompress_image(stream)
        preview = out_dir / f"{path.stem}_q{quality}_decoded.png"
        Image.fromarray(decoded).save(preview)
        print(f"  check : decoded again in {time.perf_counter() - started:.1f} s,"
              f" PSNR {psnr(image, decoded):.2f} dB, preview {preview.name}")
    return out_path


# converts a 16-bit TIFF to an 8-bit TIFF (optionally contrast-stretched) and reports the size change
def run_8bit_quant(path, out_dir=OUTPUT_DIR, stretch=STRETCH_16_TO_8):
    path = resolve_path(path)
    out_dir = resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if SUPPORTED.get(path.suffix.lower()) != "tif":
        raise ValueError(f"{path.name}: 8bit-quant expects a .tif/.tiff input")
    print(f"[8bit-quant] {path}")
    arr = load_tiff_array(path)
    if arr.dtype == np.uint8:
        print("  input is already 8-bit; rewriting it unchanged")
    elif arr.dtype != np.uint16:
        print(f"  {arr.dtype} samples: mapping their nominal range to 0..255")
    out = quantize_16_to_8(arr, stretch=stretch)
    out_path = out_dir / f"{path.stem}_8bit{path.suffix}"
    tifffile.imwrite(out_path, out, photometric="rgb" if out.ndim == 3 else "minisblack")
    in_size, out_size = path.stat().st_size, out_path.stat().st_size
    print(f"  input : {arr.shape[1]} x {arr.shape[0]} {'RGB' if arr.ndim == 3 else 'grey'} {arr.dtype},"
          f" range {arr.min()}..{arr.max()}, {fmt_bytes(in_size)}")
    print(f"  output: {out_path.name} uint8 ({'percentile-stretched' if stretch else 'full-range'} mapping),"
          f" {fmt_bytes(out_size)}, {in_size / out_size:.2f}x smaller")
    return out_path

# %% Command line
# parses the terminal arguments (falling back to the in-code settings) and runs the chosen mode
def main(argv=None):
    parser = argparse.ArgumentParser(description="JPEG-like image compression from first principles")
    parser.add_argument("path", nargs="?", default=IMAGE_PATH,
                        help=f"image file or folder of images (default: {IMAGE_PATH})")
    parser.add_argument("--mode", choices=["compress", "8bit-quant"], default=MODE,
                        help=f"compress (JPEG-like) or 8bit-quant (16-bit TIFF -> 8-bit TIFF); default {MODE}")
    parser.add_argument("--quality", type=int, default=QUALITY, help=f"1..100, default {QUALITY}")
    parser.add_argument("--out", default=OUTPUT_DIR, help=f"output folder, default {OUTPUT_DIR}")
    parser.add_argument("--stretch", action="store_true", default=STRETCH_16_TO_8,
                        help="8bit-quant: stretch the 0.1..99.9 percentile range to 0..255")
    parser.add_argument("--no-verify", action="store_true", help="compress: skip the decode check and PNG preview")
    args = parser.parse_args(argv)
    target = resolve_path(args.path)
    if target.is_dir():
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in SUPPORTED)
        if args.mode == "8bit-quant":
            files = [p for p in files if SUPPORTED[p.suffix.lower()] == "tif"]
    elif target.exists():
        files = [target]
    else:
        sys.exit(f"not found: {target}")
    if not files:
        sys.exit(f"no supported images in {target}")
    for path in files:
        if args.mode == "8bit-quant":
            run_8bit_quant(path, args.out, args.stretch)
        else:
            run_compress(path, args.quality, args.out, not args.no_verify)
        print()


if __name__ == "__main__":
    main()
