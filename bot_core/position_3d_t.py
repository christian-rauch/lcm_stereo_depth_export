"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

import bot_core.vector_3d_t

import bot_core.quaternion_t

class position_3d_t(object):
    __slots__ = ["translation", "rotation"]

    def __init__(self):
        self.translation = bot_core.vector_3d_t()
        self.rotation = bot_core.quaternion_t()

    def encode(self):
        buf = BytesIO()
        buf.write(position_3d_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        assert self.translation._get_packed_fingerprint() == bot_core.vector_3d_t._get_packed_fingerprint()
        self.translation._encode_one(buf)
        assert self.rotation._get_packed_fingerprint() == bot_core.quaternion_t._get_packed_fingerprint()
        self.rotation._encode_one(buf)

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != position_3d_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return position_3d_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = position_3d_t()
        self.translation = bot_core.vector_3d_t._decode_one(buf)
        self.rotation = bot_core.quaternion_t._decode_one(buf)
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if position_3d_t in parents: return 0
        newparents = parents + [position_3d_t]
        tmphash = (0x1275bd1ccbdaf47f+ bot_core.vector_3d_t._get_hash_recursive(newparents)+ bot_core.quaternion_t._get_hash_recursive(newparents)) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if position_3d_t._packed_fingerprint is None:
            position_3d_t._packed_fingerprint = struct.pack(">Q", position_3d_t._get_hash_recursive([]))
        return position_3d_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

