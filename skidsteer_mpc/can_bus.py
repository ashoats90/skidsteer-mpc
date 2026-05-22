"""CAN output layer (Classic CAN 2.0). E2E-Profile-2-style protection on the
safety-relevant command frame: CRC8/AUTOSAR + 4-bit rolling counter, so the drive
detects corrupted / lost / duplicated / stale frames and fails safe locally.

NOTE: E2E layout follows the *pattern* of AUTOSAR E2E Profile 2 (CRC byte 0,
counter in low nibble of byte 1, Data-ID folded into the CRC seed); it is
simplified, not bit-exact AUTOSAR. Match it to your drive's DBC before integration."""
from __future__ import annotations
from dataclasses import dataclass
from .config import CanConfig

# status-flag bit positions (byte 6 of TRACK_CMD)
FLAG_SOLVER_OK = 1 << 0
FLAG_TIMEOUT   = 1 << 1
FLAG_BUFFERED  = 1 << 2
FLAG_CLAMPED   = 1 << 3


def crc8_autosar(data: bytes) -> int:
    """CRC-8/AUTOSAR: poly 0x2F, init 0xFF, xorout 0xFF (check value 0xDF)."""
    crc = 0xFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x2F) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc ^ 0xFF


@dataclass
class TrackCmd:
    v_left: float
    v_right: float
    mode: int
    counter: int
    flags: int


class TrackCmdCodec:
    """Encodes/decodes the TRACK_CMD frame with E2E protection."""
    def __init__(self, cfg: CanConfig | None = None):
        self.cfg = cfg or CanConfig()

    def _i16(self, v_mps: float) -> int:
        raw = int(round(v_mps / self.cfg.v_scale))
        return max(-32768, min(32767, raw))

    def _crc(self, payload: bytes) -> int:
        return crc8_autosar(bytes([self.cfg.data_id]) + payload[1:8])

    def encode(self, v_left, v_right, mode, counter, flags=0) -> bytes:
        p = bytearray(8)
        p[1] = ((mode & 0x0F) << 4) | (counter & 0x0F)
        p[2:4] = int(self._i16(v_left)).to_bytes(2, "little", signed=True)
        p[4:6] = int(self._i16(v_right)).to_bytes(2, "little", signed=True)
        p[6] = flags & 0xFF
        p[0] = self._crc(bytes(p))
        return bytes(p)

    def decode(self, frame: bytes) -> TrackCmd:
        return TrackCmd(
            v_left=int.from_bytes(frame[2:4], "little", signed=True) * self.cfg.v_scale,
            v_right=int.from_bytes(frame[4:6], "little", signed=True) * self.cfg.v_scale,
            mode=(frame[1] >> 4) & 0x0F, counter=frame[1] & 0x0F, flags=frame[6])

    def crc_ok(self, frame: bytes) -> bool:
        return self._crc(frame) == frame[0]


class CmdReceiver:
    """Drive-side validation: CRC + counter continuity + freshness."""
    def __init__(self, codec: TrackCmdCodec | None = None):
        self.codec = codec or TrackCmdCodec()
        self.last_counter = None
        self.last_rx_time = None

    def validate(self, frame: bytes, now_s: float):
        if not self.codec.crc_ok(frame):
            return False, "crc_mismatch", None
        cmd = self.codec.decode(frame)
        if self.last_counter is not None:
            if cmd.counter == self.last_counter:
                return False, "repeated_frame", cmd
            if cmd.counter != (self.last_counter + 1) & 0x0F:
                return False, "counter_gap", cmd
        if (self.last_rx_time is not None
                and (now_s - self.last_rx_time) > self.codec.cfg.rx_timeout_s):
            self.last_counter, self.last_rx_time = cmd.counter, now_s
            return False, "stale", cmd
        self.last_counter, self.last_rx_time = cmd.counter, now_s
        return True, "ok", cmd


class MpcStatusCodec:
    """Diagnostic status frame (not E2E-protected)."""
    def encode(self, solve_time_s, fault_code, consec_fault, track_err_cm, mode) -> bytes:
        st = max(0, min(65535, int(round(solve_time_s * 1000 / 0.05))))
        te = max(0, min(65535, int(round(track_err_cm / 0.1))))
        out = bytearray(8)
        out[0:2] = st.to_bytes(2, "little")
        out[2] = fault_code & 0xFF
        out[3] = min(255, consec_fault) & 0xFF
        out[4:6] = te.to_bytes(2, "little")
        out[6] = mode & 0xFF
        return bytes(out)
