"""Canonical model classes for inference and training."""
from .detector import SpikeDetector
from .huidurep.CMAES import CMAES

VanillaDet = SpikeDetector
HuiduRep = CMAES

__all__ = ["SpikeDetector", "VanillaDet", "CMAES", "HuiduRep"]
