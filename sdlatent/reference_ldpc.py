from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceLdpcMode:
    name: str
    modulation: str
    code_rate: str
    required_esn0_db: float
    note: str


DVB_S2_QPSK_1_4 = ReferenceLdpcMode(
    name="DVB-S2 QPSK 1/4 LDPC+BCH",
    modulation="QPSK",
    code_rate="1/4",
    required_esn0_db=-2.35,
    note="ETSI EN 302 307 Annex H/Table 13 style QEF AWGN requirement for the most robust DVB-S2 MODCOD.",
)


class PublishedThresholdLDPC:
    """A literature-backed conservative LDPC metadata baseline.

    This baseline treats the metadata header as decodable if the channel-symbol
    SNR is above the published QEF threshold for a very robust LDPC+BCH mode.
    It intentionally gives conventional coding the benefit of a long, optimized
    standard code rather than the small toy LDPC used in early exploration.
    """

    name = "published_threshold_ldpc"

    def __init__(self, mode: ReferenceLdpcMode = DVB_S2_QPSK_1_4, margin_db: float = 0.0):
        self.mode = mode
        self.margin_db = float(margin_db)

    @property
    def threshold_db(self) -> float:
        return self.mode.required_esn0_db + self.margin_db

    def decode_success(self, snr_db: float) -> bool:
        return float(snr_db) >= self.threshold_db
