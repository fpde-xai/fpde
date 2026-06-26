"""Result types for RawFeat Dynamic-FPDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np


DynamicMethod = Literal["diff", "cos", "hyb"]


@dataclass(frozen=True)
class DynamicFPDEResult:
    """Explanation result returned by :class:`DynamicFPDEEngine`.

    The attribution matrix uses the concatenated RawFeat representation:
    raw channels first, optional extracted features next, and optional ``dt``
    last. Convenience views expose the channel groups separately.
    """

    method: DynamicMethod
    target_class: Any
    rival_class: Any
    evidence: float
    attributions: np.ndarray
    raw_attributions: np.ndarray
    feature_attributions: Optional[np.ndarray]
    dt_attributions: Optional[np.ndarray]
    time_attributions: np.ndarray
    group_attributions: Dict[str, float]
    mask: np.ndarray
    feature_slices: Dict[str, slice]
    audit: Dict[str, Any]


__all__ = ["DynamicFPDEResult", "DynamicMethod"]
