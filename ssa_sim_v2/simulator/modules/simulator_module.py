# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict, Callable

from collections import namedtuple

import numpy as np

# ------------------------------------------------------------


class SimulatorModule(object):
    """
    Base class for all simulator modules.

    :ivar int seed: Seed for the random number generator.
    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with params for every segment.
    :ivar Dict[tuple, Callable] segment_func_map: Dict with functions
        returning constant probability for every bid for every segment.
    """

    Params = namedtuple('Params', [])

    def __init__(self, prior={(0,): Params()}, seed=9):
        """
        :param dict prior: Dict with params for every segment.
        :param int seed: Seed for the random number generator.
        """

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.prior = prior

        self.segment_func_map = None  # type: Dict[tuple, Callable]
