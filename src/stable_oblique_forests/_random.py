"""Create local NumPy generators without changing global random state."""
from __future__ import annotations

import numpy as np


def make_rng(random_state):
    """Accept None, integer seeds, Generator and legacy sklearn RandomState."""
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(int(random_state.randint(0, np.iinfo(np.uint32).max)))
    return np.random.default_rng(random_state)
