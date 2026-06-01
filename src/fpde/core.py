"""Backward-compatible aggregate module for the FPDE public API."""

from .types import *  # noqa: F401,F403
from .metrics import *  # noqa: F401,F403
from .prototypes import *  # noqa: F401,F403
from .explainers import *  # noqa: F401,F403
from .selection import *  # noqa: F401,F403
from .engine import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
from .plotting import *  # noqa: F401,F403
from ._batch import _full_topk_mask, _topk_masks_for_counts  # noqa: F401

from .types import __all__ as _types_all
from .metrics import __all__ as _metrics_all
from .prototypes import __all__ as _prototypes_all
from .explainers import __all__ as _explainers_all
from .selection import __all__ as _selection_all
from .engine import __all__ as _engine_all
from .utils import __all__ as _utils_all
from .plotting import __all__ as _plotting_all

__all__ = list(dict.fromkeys(
    [
        *_types_all,
        *_metrics_all,
        *_prototypes_all,
        *_explainers_all,
        *_selection_all,
        *_engine_all,
        *_utils_all,
        *_plotting_all,
    ]
))
