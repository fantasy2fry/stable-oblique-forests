"""Create local NumPy generators without changing global random state."""
from __future__ import annotations

import numpy as np


def make_rng(random_state):
    """Accept None, integer seeds, Generator and legacy sklearn RandomState."""
    if isinstance(random_state, np.random.RandomState):
        # Explicit dtype: the default is the C long, which is 32-bit on Windows and
        # would reject a uint32-sized upper bound. int32 max is valid everywhere.
        seed = random_state.randint(0, np.iinfo(np.int32).max, dtype=np.int64)
        return np.random.default_rng(int(seed))
    return np.random.default_rng(random_state)