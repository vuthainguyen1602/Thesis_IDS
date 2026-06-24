"""Backward-compatible facade for the IDS thesis utilities.

The implementation now lives in the `idslib` package, split by concern
(core, data, modeling, viz). Every `from shared_utils import X` used by the
ml_*.py experiment scripts keeps working unchanged.
"""
from idslib.core import *
from idslib.data import *
from idslib.modeling import *
from idslib.viz import *

# Underscore-prefixed names are not pulled in by `import *`; re-export the ones
# that scripts (or callers) reference so the public API stays identical.
from idslib.core import (
    _DEFAULT_DATA_DIR, _REPRO_ENV_KEYS, _SPARK_MASTER,
    _allow_local_spark, _configure_java_home, _is_arm64,
)
from idslib.data import _clean_name, _leaky_port_cols
from idslib.modeling import _get_param, _get_best_params, _t_critical_95
