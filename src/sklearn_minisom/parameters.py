"""Collection of parameters used for SOM initialization."""

from enum import StrEnum


class DecayFunction(StrEnum):
    InverseDecayToZero = "inverse_decay_to_zero"
    LinearDecayToZero = "linear_decay_to_zero"
    AsymptoticDecay = "asymptotic_decay"


class NeighborhoodFunction(StrEnum):
    Gaussian = "gaussian"
    MexicanHat = "mexican_hat"
    Bubble = "bubble"
    Triangle = "triangle"


class Topology(StrEnum):
    Rectangular = "rectangular"
    Hexagonal = "hexagonal"


class ActivationDistance(StrEnum):
    Euclidean = "euclidean"
    Cosine = "cosine"
    Manhattan = "manhattan"
    Chebyshev = "chebyshev"


class SigmaDecayFunction(StrEnum):
    InverseDecayToOne = "inverse_decay_to_one"
    LinearDecayToOne = "linear_decay_to_one"
    AsymptoticDecay = "asymptotic_decay"
