"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class gps_data_t(object):
    __slots__ = ["utime", "gps_lock", "longitude", "latitude", "elev", "horizontal_accuracy", "vertical_accuracy", "numSatellites", "speed", "heading", "xyz_pos", "gps_time"]

    def __init__(self):
        self.utime = 0
        self.gps_lock = 0
        self.longitude = 0.0
        self.latitude = 0.0
        self.elev = 0.0
        self.horizontal_accuracy = 0.0
        self.vertical_accuracy = 0.0
        self.numSatellites = 0
        self.speed = 0.0
        self.heading = 0.0
        self.xyz_pos = [ 0.0 for dim0 in range(3) ]
        self.gps_time = 0.0

    def encode(self):
        buf = BytesIO()
        buf.write(gps_data_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">qidddddidd", self.utime, self.gps_lock, self.longitude, self.latitude, self.elev, self.horizontal_accuracy, self.vertical_accuracy, self.numSatellites, self.speed, self.heading))
        buf.write(struct.pack('>3d', *self.xyz_pos[:3]))
        buf.write(struct.pack(">d", self.gps_time))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != gps_data_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return gps_data_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = gps_data_t()
        self.utime, self.gps_lock, self.longitude, self.latitude, self.elev, self.horizontal_accuracy, self.vertical_accuracy, self.numSatellites, self.speed, self.heading = struct.unpack(">qidddddidd", buf.read(72))
        self.xyz_pos = struct.unpack('>3d', buf.read(24))
        self.gps_time = struct.unpack(">d", buf.read(8))[0]
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if gps_data_t in parents: return 0
        tmphash = (0x6be9070f34520a8b) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if gps_data_t._packed_fingerprint is None:
            gps_data_t._packed_fingerprint = struct.pack(">Q", gps_data_t._get_hash_recursive([]))
        return gps_data_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

