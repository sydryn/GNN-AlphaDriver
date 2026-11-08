from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)

# GRCh38 primary chromosome lengths, in chromosome order.
CHROMOSOME_LENGTHS = dict(zip(
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"],
    (248956422, 242193529, 198295559, 190214555, 181538259, 170805979,
     159345973, 145138636, 138394717, 133797422, 135086622, 133275309,
     114364328, 107043718, 101991189, 90338345, 83257441, 80373285,
     58617616, 64444167, 46709983, 50818468, 156040895, 57227415),
    strict=True,
))
WINDOWS = {"100kb": 131_072, "1mb": 1_048_576}


@dataclass(frozen=True, slots=True)
class VariantKey:
    chromosome: str
    position: int
    reference: str
    alternate: str

    def __post_init__(self) -> None:
        chrom = "chr" + self.chromosome.strip().lower().removeprefix("chr").upper()
        ref, alt = self.reference.strip().upper(), self.alternate.strip().upper()
        if chrom not in CHROMOSOME_LENGTHS:
            raise ValueError("Choose a GRCh38 primary chromosome (1–22, X, or Y).")
        if type(self.position) is not int or not 1 <= self.position <= CHROMOSOME_LENGTHS[chrom]:
            raise ValueError("Position must be a 1-based coordinate within the chromosome.")
        if ref not in {"A", "C", "G", "T"} or alt not in {"A", "C", "G", "T"} or ref == alt:
            raise ValueError("REF and ALT must be distinct single bases (A, C, G, or T).")
        object.__setattr__(self, "chromosome", chrom)
        object.__setattr__(self, "reference", ref)
        object.__setattr__(self, "alternate", alt)

    def __str__(self) -> str:
        return f"{self.chromosome}:{self.position}:{self.reference}>{self.alternate}"


@dataclass(frozen=True, slots=True)
class GenomicInterval:
    chromosome: str
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start

    @classmethod
    def around(cls, variant: VariantKey, window: str) -> "GenomicInterval":
        if window not in WINDOWS:
            raise ValueError("Window must be 100kb or 1mb.")
        width = WINDOWS[window]
        start = min(max(0, variant.position - 1 - width // 2),
                    CHROMOSOME_LENGTHS[variant.chromosome] - width)
        return cls(variant.chromosome, start, start + width)


@dataclass(frozen=True, slots=True)
class GenomicArray:
    interval: GenomicInterval
    resolution: int
    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float32)
        if self.resolution <= 0 or self.interval.width <= 0:
            raise ValueError("Track width and resolution must be positive.")
        if values.ndim != 1 or values.size * self.resolution != self.interval.width:
            raise ValueError("Track must be 1D and cover its entire interval.")
        if not np.isfinite(values).all():
            raise ValueError("Track values must be finite.")
        object.__setattr__(self, "values", values)

    def delta_from(self, reference: "GenomicArray") -> "GenomicArray":
        if self.interval != reference.interval or self.resolution != reference.resolution:
            raise ValueError("REF and ALT tracks must have aligned bins.")
        return GenomicArray(self.interval, self.resolution, self.values - reference.values)


class VariantHashMap(Generic[K, T]):
    """Index variants or graph node identifiers by a hashable key."""

    def __init__(self) -> None:
        self._records: dict[K, T] = {}

    def add(self, key: K, value: T) -> None:
        if key in self._records:
            raise ValueError(f"Duplicate key: {key}")
        self._records[key] = value

    def __getitem__(self, key: K) -> T:
        return self._records[key]


class SparseCOO:
    def __init__(self, source_count: int, target_count: int) -> None:
        if min(source_count, target_count) < 0:
            raise ValueError("Node counts cannot be negative.")
        self.source_count = source_count
        self.target_count = target_count
        self._edges: dict[tuple[int, int], None] = {}

    def add(self, source: int, target: int) -> None:
        if not (0 <= source < self.source_count and 0 <= target < self.target_count):
            raise ValueError("Edge index lies outside its node type.")
        self._edges[source, target] = None

    def to_array(self) -> NDArray[np.int64]:
        return np.asarray(list(self._edges), dtype=np.int64).reshape(-1, 2).T.copy()
