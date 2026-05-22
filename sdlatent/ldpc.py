from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _int_to_bits(value: int, width: int) -> np.ndarray:
    return np.array([(value >> i) & 1 for i in range(width - 1, -1, -1)], dtype=np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def crc8(bits: np.ndarray, poly: int = 0x07, init: int = 0x00) -> int:
    """CRC-8 over a bit stream, MSB first."""
    crc = init
    for bit in bits.astype(np.uint8):
        crc ^= int(bit) << 7
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def shape_to_payload(shape: tuple[int, int, int]) -> np.ndarray:
    h, w, c = shape
    if max(shape) >= 256:
        raise ValueError(f"shape fields must fit in 8 bits, got {shape}")
    fields = np.concatenate([_int_to_bits(h, 8), _int_to_bits(w, 8), _int_to_bits(c, 8)])
    checksum = _int_to_bits(crc8(fields), 8)
    return np.concatenate([fields, checksum]).astype(np.uint8)


def payload_to_shape(bits: np.ndarray) -> tuple[int, int, int] | None:
    bits = np.asarray(bits[:32], dtype=np.uint8)
    fields, checksum = bits[:24], bits[24:32]
    if crc8(fields) != _bits_to_int(checksum):
        return None
    return (_bits_to_int(fields[:8]), _bits_to_int(fields[8:16]), _bits_to_int(fields[16:24]))


@dataclass
class LdpcDecode:
    shape: tuple[int, int, int] | None
    crc_ok: bool
    syndrome_ok: bool
    iterations: int


class SystematicLDPC:
    """A compact systematic LDPC-like code for metadata baselines.

    The parity check is H = [A | I]. Encoding is systematic:
    codeword = [payload, A payload]. Decoding uses normalized min-sum.
    This is intentionally small and self-contained so the experiment does not
    depend on external LDPC packages.
    """

    def __init__(
        self,
        k: int = 32,
        n: int = 192,
        column_weight: int = 4,
        seed: int = 3102026,
        min_sum_scale: float = 0.75,
        max_iter: int = 50,
    ):
        if n <= k:
            raise ValueError("n must be greater than k")
        self.k = int(k)
        self.n = int(n)
        self.m = self.n - self.k
        self.column_weight = int(column_weight)
        self.min_sum_scale = float(min_sum_scale)
        self.max_iter = int(max_iter)
        rng = np.random.default_rng(seed)
        self.A = np.zeros((self.m, self.k), dtype=np.uint8)
        for col in range(self.k):
            rows = rng.choice(self.m, size=self.column_weight, replace=False)
            self.A[rows, col] = 1
        self.var_to_checks: list[np.ndarray] = []
        self.check_to_vars: list[np.ndarray] = []
        self.var_to_edges: list[np.ndarray] = []
        self.check_to_edges: list[np.ndarray] = []
        edge_checks = []
        edge_vars = []
        for var in range(self.n):
            if var < self.k:
                checks = np.flatnonzero(self.A[:, var])
            else:
                checks = np.array([var - self.k], dtype=np.int64)
            self.var_to_checks.append(checks.astype(np.int64))
        for check in range(self.m):
            payload_vars = np.flatnonzero(self.A[check])
            self.check_to_vars.append(np.concatenate([payload_vars, np.array([self.k + check])]).astype(np.int64))
        for check, vars_ in enumerate(self.check_to_vars):
            check_edges = []
            for var in vars_:
                edge_checks.append(check)
                edge_vars.append(int(var))
                check_edges.append(len(edge_vars) - 1)
            self.check_to_edges.append(np.array(check_edges, dtype=np.int64))
        edge_vars_arr = np.array(edge_vars, dtype=np.int64)
        for var in range(self.n):
            self.var_to_edges.append(np.flatnonzero(edge_vars_arr == var).astype(np.int64))
        self.edge_check = np.array(edge_checks, dtype=np.int64)
        self.edge_var = edge_vars_arr

    @property
    def rate(self) -> float:
        return self.k / self.n

    def encode(self, payload: np.ndarray) -> np.ndarray:
        payload = np.asarray(payload[: self.k], dtype=np.uint8)
        parity = (self.A @ payload) & 1
        return np.concatenate([payload, parity.astype(np.uint8)])

    def syndrome_ok(self, codeword: np.ndarray) -> bool:
        codeword = np.asarray(codeword[: self.n], dtype=np.uint8)
        payload = codeword[: self.k]
        parity = codeword[self.k :]
        syndrome = ((self.A @ payload) & 1) ^ parity
        return bool(np.all(syndrome == 0))

    def decode_llr(self, llr: np.ndarray) -> tuple[np.ndarray, bool, int]:
        llr = np.asarray(llr[: self.n], dtype=np.float64)
        var_msg = llr[self.edge_var].copy()
        check_msg = np.zeros_like(var_msg)
        hard = np.zeros(self.n, dtype=np.uint8)
        for it in range(1, self.max_iter + 1):
            for edges in self.check_to_edges:
                incoming = var_msg[edges]
                signs = np.where(incoming >= 0.0, 1.0, -1.0)
                abs_vals = np.abs(incoming)
                total_sign = float(np.prod(signs))
                min1_idx = int(np.argmin(abs_vals))
                min1 = float(abs_vals[min1_idx])
                if len(abs_vals) > 1:
                    min2 = float(np.partition(abs_vals, 1)[1])
                else:
                    min2 = min1
                for idx, edge in enumerate(edges):
                    mag = min2 if idx == min1_idx else min1
                    msg = self.min_sum_scale * total_sign * signs[idx] * mag
                    check_msg[edge] = float(msg)

            posterior = llr + np.bincount(self.edge_var, weights=check_msg, minlength=self.n)
            hard = (posterior < 0.0).astype(np.uint8)
            if self.syndrome_ok(hard):
                return hard[: self.k], True, it

            var_msg = posterior[self.edge_var] - check_msg

        return hard[: self.k], False, self.max_iter


class LDPCShapeHeader:
    """Transmit shape metadata through a separate LDPC-coded BPSK header."""

    name = "ldpc_shape_header"

    def __init__(self, n: int = 192, seed: int = 3102026, max_iter: int = 50):
        self.code = SystematicLDPC(k=32, n=n, seed=seed, max_iter=max_iter)

    @property
    def overhead_symbols(self) -> int:
        return self.code.n

    @property
    def rate(self) -> float:
        return self.code.rate

    def transmit_decode(self, shape: tuple[int, int, int], snr_db: float, rng: np.random.Generator) -> LdpcDecode:
        payload = shape_to_payload(shape)
        codeword = self.code.encode(payload)
        symbols = 1.0 - 2.0 * codeword.astype(np.float64)
        noise_var = 1.0 / (10.0 ** (snr_db / 10.0))
        rx = symbols + rng.normal(0.0, np.sqrt(noise_var), size=symbols.shape)
        llr = 2.0 * rx / noise_var
        decoded_payload, syndrome_ok, iterations = self.code.decode_llr(llr)
        decoded_shape = payload_to_shape(decoded_payload)
        return LdpcDecode(decoded_shape, decoded_shape is not None, syndrome_ok, iterations)
