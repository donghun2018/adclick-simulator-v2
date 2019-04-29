# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict

from collections import namedtuple

import numpy as np

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------


class MultiStateSimulatorModule(object):
    """
    Base class for all multi state simulator modules.

    :ivar int seed: Seed for the random number generator.
    :ivar np.random.RandomState rng: Random number generator.
    :ivar pd.DataFrame priors: DataFrame with columns for the state
        (e.g. date, hour_of_week) and the prior column. The last column
        defines priors (in the form of dictionaries) for every state.
    :ivar pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines class for for given dates and hours of week.
    :ivar Dict[str, SimulatorModule] models: Dictionary of single state modules for every valid pair
        of date and hour of week.
    """

    Params = namedtuple('Params', [])

    def __init__(self, priors=None, base_classes=None, seed=9):
        """
        :param pd.DataFrame priors: DataFrame with columns for the state
            (e.g. date, hour_of_week) and the prior column. The last column
            defines priors (in the form of dictionaries) for every state.
        :param pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines class for given dates and hours of week.
        :param int seed: Seed for the random number generator.
        """

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.priors = priors
        self.base_classes = base_classes

        seed_min = 100000
        seed_max = 999999
        seeds = self.rng.randint(low=seed_min, high=seed_max, size=len(self.priors))

        base_df = priors.copy()
        base_df.loc[:, "base_class"] = base_classes["base_class"]
        base_df.loc[:, "seed"] = seeds

        self.models = {}  # type: Dict[str, SimulatorModule]

        for index, row in base_df.iterrows():
            self.models["{}.{}".format(row["date"], row["hour_of_week"])] = row["base_class"](row["prior"], row["seed"])

    def get_models(self):
        """
        Returns a dictionary of underlying click probability models.

        :return: Dictionary of models.
        :rtype: Dict[str, SimulatorModule]
        """
        return self.models
