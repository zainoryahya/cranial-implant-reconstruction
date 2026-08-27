"""
Minimal pure-Python DICOM parser (no pydicom / gdcm dependency).
Handles uncompressed Implicit VR Little Endian and Explicit VR Little Endian
transfer syntaxes, which cover standard CT/MR exports from PACS.
"""
import struct
import numpy as np

# VRs that use the "long" explicit form: tag(4) + VR(2) + reserved(2) + length(4)
LONG_VRS = {b'OB', b'OW', b'OF', b'SQ', b'UT', b'UN', b'OD', b'OL', b'UC', b'UR'}
# VRs whose value we care to decode as text/number
TEXT_VRS = {b'AE', b'AS', b'CS', b'DA', b'DS', b'DT', b'IS', b'LO', b'LT', b'PN',
            b'SH', b'ST', b'TM', b'UI', b'UC', b'UR'}

def _read_tag(buf, pos):
    group, elem = struct.unpack_from('<HH', buf, pos)
    return group, elem, pos + 4

def parse_dicom(path, stop_before_pixels=False):
    """Return dict of tag->value plus 'PixelData' bytes (unless stop_before_pixels)."""
    with open(path, 'rb') as f:
        data = f.read()

    assert data[128:132] == b'DICM', f"Not a DICOM file (no DICM magic): {path}"
    pos = 132
    tags = {}

    # --- File Meta Info (group 0002) is ALWAYS Explicit VR Little Endian ---
    group, elem, p2 = _read_tag(data, pos)
    while group == 0x0002:
        vr = data[p2:p2+2]
        if vr in LONG_VRS:
            length = struct.unpack_from('<I', data, p2+4)[0]
            vpos = p2 + 8
        else:
            length = struct.unpack_from('<H', data, p2+2)[0]
            vpos = p2 + 4
        value = data[vpos:vpos+length]
        tags[(group, elem)] = value
        pos = vpos + length
        if pos >= len(data):
            break
        group, elem, p2 = _read_tag(data, pos)

    ts_uid = tags.get((0x0002, 0x0010), b'').rstrip(b'\x00 ').decode('ascii', 'ignore')
    explicit = True
    if ts_uid == '1.2.840.10008.1.2':
        explicit = False  # Implicit VR Little Endian
    # (we don't support Big Endian or compressed transfer syntaxes here)

    pixel_data = None
    while pos < len(data) - 8:
        group, elem, p2 = _read_tag(data, pos)
        if explicit:
            vr = data[p2:p2+2]
            if vr in LONG_VRS:
                length = struct.unpack_from('<I', data, p2+4)[0]
                vpos = p2 + 8
            else:
                length = struct.unpack_from('<H', data, p2+2)[0]
                vpos = p2 + 4
        else:
            vr = None
            length = struct.unpack_from('<I', data, p2)[0]
            vpos = p2 + 4

        if length == 0xFFFFFFFF:
            # undefined length (sequences / encapsulated pixel data) - not supported, bail
            break

        if group == 0x7FE0 and elem == 0x0010:
            if stop_before_pixels:
                pixel_data = None
                break
            pixel_data = data[vpos:vpos+length]
            pos = vpos + length
            continue

        value = data[vpos:vpos+length]
        tags[(group, elem)] = value
        pos = vpos + length

    tags['PixelData'] = pixel_data
    return tags

# VR of the specific tags we care about (needed because Implicit VR Little
# Endian doesn't carry VR in the stream at all, so we must know it a priori).
TAG_VR = {
    (0x0028, 0x0010): 'US',  # Rows
    (0x0028, 0x0011): 'US',  # Columns
    (0x0028, 0x0100): 'US',  # BitsAllocated
    (0x0028, 0x0101): 'US',  # BitsStored
    (0x0028, 0x0103): 'US',  # PixelRepresentation
    (0x0028, 0x0002): 'US',  # SamplesPerPixel
    (0x0028, 0x0030): 'DS',  # PixelSpacing
    (0x0028, 0x1052): 'DS',  # RescaleIntercept
    (0x0028, 0x1053): 'DS',  # RescaleSlope
    (0x0018, 0x0050): 'DS',  # SliceThickness
    (0x0018, 0x0088): 'DS',  # SpacingBetweenSlices
    (0x0020, 0x0032): 'DS',  # ImagePositionPatient
    (0x0020, 0x0013): 'IS',  # InstanceNumber
    (0x0008, 0x0060): 'CS',  # Modality
    (0x0008, 0x0016): 'UI',  # SOPClassUID
}

def _txt(tags, group, elem, default=None):
    v = tags.get((group, elem))
    if v is None:
        return default
    return v.rstrip(b'\x00 ').decode('ascii', 'ignore').strip()

def _raw_us_list(raw):
    n = len(raw) // 2
    return list(struct.unpack_from('<%dH' % n, raw, 0))

def _nums(tags, group, elem, default=None):
    raw = tags.get((group, elem))
    if raw is None:
        return default
    vr = TAG_VR.get((group, elem), 'DS')
    if vr == 'US':
        return [float(x) for x in _raw_us_list(raw)]
    s = raw.rstrip(b'\x00 ').decode('ascii', 'ignore').strip()
    if not s:
        return default
    return [float(x) for x in s.split('\\') if x != '']

def _int(tags, group, elem, default=None):
    raw = tags.get((group, elem))
    if raw is None:
        return default
    vr = TAG_VR.get((group, elem), 'IS')
    if vr == 'US':
        return _raw_us_list(raw)[0]
    s = raw.rstrip(b'\x00 ').decode('ascii', 'ignore').strip()
    if s == '':
        return default
    return int(float(s))

class Slice:
    def __init__(self, path):
        t = parse_dicom(path)
        self.path = path
        self.rows = _int(t, 0x0028, 0x0010)
        self.cols = _int(t, 0x0028, 0x0011)
        self.bits_allocated = _int(t, 0x0028, 0x0100, 16)
        self.pixel_rep = _int(t, 0x0028, 0x0103, 0)  # 0=unsigned,1=signed
        self.rescale_slope = (_nums(t, 0x0028, 0x1053) or [1.0])[0]
        self.rescale_intercept = (_nums(t, 0x0028, 0x1052) or [0.0])[0]
        ps = _nums(t, 0x0028, 0x0030) or [1.0, 1.0]
        self.pixel_spacing = ps  # [row spacing, col spacing]
        ipp = _nums(t, 0x0020, 0x0032)
        self.image_position = ipp  # x,y,z
        self.instance_number = _int(t, 0x0020, 0x0013, 0)
        self.slice_thickness = (_nums(t, 0x0018, 0x0050) or [None])[0]
        self.spacing_between = (_nums(t, 0x0018, 0x0088) or [None])[0]
        self.modality = _txt(t, 0x0008, 0x0060)
        self.sop_class = _txt(t, 0x0008, 0x0016)
        raw = t['PixelData']
        dtype = np.uint16 if self.pixel_rep == 0 else np.int16
        if self.bits_allocated == 8:
            dtype = np.uint8 if self.pixel_rep == 0 else np.int8
        n = self.rows * self.cols
        arr = np.frombuffer(raw, dtype=dtype, count=n).reshape(self.rows, self.cols)
        self.pixels_raw = arr
        self.hu = arr.astype(np.float32) * self.rescale_slope + self.rescale_intercept

    def z(self):
        if self.image_position:
            return self.image_position[2]
        return float(self.instance_number)
