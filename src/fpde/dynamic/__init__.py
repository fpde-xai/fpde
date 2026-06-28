"""Dynamic-FPDE public API."""

from .legacy import *  # noqa: F401,F403
from .legacy import _labels_unique, _validate_eps, _validate_lambda_hyb
from .types import *  # noqa: F401,F403
from .preprocessing import *  # noqa: F401,F403
from .engine import *  # noqa: F401,F403
from .generator import *  # noqa: F401,F403
from .bayesian import *  # noqa: F401,F403

from .legacy import __all__ as _legacy_all
from .types import __all__ as _types_all
from .preprocessing import __all__ as _preprocessing_all
from .engine import __all__ as _engine_all
from .generator import __all__ as _generator_all
from .bayesian import __all__ as _bayesian_all

__all__ = list(
    dict.fromkeys(
        [
            *_legacy_all,
            *_types_all,
            *_preprocessing_all,
            *_engine_all,
            *_generator_all,
            *_bayesian_all,
        ]
    )
)
