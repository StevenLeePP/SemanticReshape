"""Self-describing latent shape experiments for semantic communication."""

from .channel import awgn, signal_power
from .latent_source import ShapeSpec, make_default_shape_book, sample_latent
from .shape_code import (
    CompactDigitalHeader,
    PrefixBpskHeader,
    ProjectionQIMID,
    SpreadSpectrumShapeFields,
    SpreadSpectrumShapeComponents,
    RobustSpreadSpectrumShapeComponents,
    SpreadSpectrumID,
)
from .semantic_header import SemanticShapeHeader
from .redundant_header import RepetitionCodedShapeHeader, RandomCodebookShapeHeader

__all__ = [
    "awgn",
    "signal_power",
    "ShapeSpec",
    "make_default_shape_book",
    "sample_latent",
    "CompactDigitalHeader",
    "PrefixBpskHeader",
    "ProjectionQIMID",
    "SpreadSpectrumShapeFields",
    "SpreadSpectrumShapeComponents",
    "RobustSpreadSpectrumShapeComponents",
    "SpreadSpectrumID",
    "SemanticShapeHeader",
    "RepetitionCodedShapeHeader",
    "RandomCodebookShapeHeader",
]
