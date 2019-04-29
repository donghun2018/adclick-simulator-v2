# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":

    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from abc import abstractmethod
from typing import Dict
from collections import namedtuple

import numpy as np

import ssa_sim_v2.tools.dict_utils as dict_utils
from ssa_sim_v2.simulator.modules.commons.noise import NormalNoiseGenerator

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------


class AveragePositionModule(SimulatorModule):
    """
    Basic module for the average position in AdWords results based on the max click probability
    and the actual probability.

    A normal noise can be added to this value.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps segments with functions calculating average_position.
    """

    Params = namedtuple('Params', ['max_cp', 'noise_level', 'noise_type'])
    """
    :param float max_cp: Maximal click probability.
    :param float noise_level: Noise level.
    :param float noise_type: Noise type:

        * additive -- normal noise with noise_level as the standard deviation,
        * multiplicative -- normal noise with noise_level as the standard deviation times the base value.
    """

    def __init__(self, prior={(0,): Params(1.0, 0.0, "multiplicative")}, seed=123):
        """
        :param Dict[tuple, AveragePositionHyperbolicSegments.Params] prior: Dict
            with prior values for the module.
        :param int seed: Seed for the random number generator.
        """
        super().__init__(prior, seed)

        self.max_position = 10.0

        self.segment_func_map = dict_utils.dict_apply(prior, self.generate_average_position_func)

    @abstractmethod
    def generate_average_position_func(self, params):
        """
        :param AveragePositionHyperbolicSegments.Params params: Params.
        """
        del params

        def get_average_position(cp):
            del cp
            return 1.0

        return get_average_position

    def get_average_position(self, p, attr=(0,)):
        """
        Returns average position from a click probability assuming a hyperbolic model.

        :return: Average position from a click probability.
        :rtype: float
        """
        return self.segment_func_map[attr](p)


class AveragePositionHyperbolicModule(AveragePositionModule):
    """
    Basic module for the average position in AdWords results based on the max click probability
    and the actual probability.
    
    A normal noise can be added to this value.
    
    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps segments with functions calculating average_position.
    """

    Params = namedtuple('Params', ['max_cp', 'noise_level', 'noise_type'])
    """
    :param float max_cp: Maximal click probability.
    :param float noise_level: Noise level.
    :param float noise_type: Noise type:

        * additive -- normal noise with noise_level as the standard deviation,
        * multiplicative -- normal noise with noise_level as the standard deviation times the base value.
    """

    def generate_average_position_func(self, params):
        """
        :param AveragePositionHyperbolicSegments.Params params: Params.
        """

        def get_average_position(cp):
            avg_pos = min(params.max_cp / max(cp, 0.0001), self.max_position)

            avg_pos = NormalNoiseGenerator(params.noise_type).generate_value_with_noise(
                avg_pos, params.noise_level, self.rng)

            return max(round(avg_pos, 2), 1.0)

        return get_average_position


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":

    import unittest


    class TestAveragePositionHyperbolicModule(unittest.TestCase):
        def test_sanity(self):
            print("AveragePositionHyperbolicModule class sample run -------------")

            segments = [(seg1, seg2) for seg1 in [0, 1] for seg2 in [0, 1]]
            max_cp_priors = np.random.uniform(low=0.1, high=0.4, size=len(segments))
            noise_levels = [0, 0.1, 0.1, 0.3]
            noise_types = ["multiplicative", "multiplicative", "additive", "multiplicative", ]

            prior = {segment: AveragePositionHyperbolicModule.Params(max_cp_prior, noise_level, noise_type) for
                     segment, max_cp_prior, noise_level, noise_type in
                     zip(segments, max_cp_priors, noise_levels, noise_types)}

            max_cp_model = AveragePositionHyperbolicModule(prior)

            p = 0.1

            reps = 1
            data = []
            for segment in segments:
                for rep in range(reps):
                    out = []
                    max_cp = max_cp_model.get_average_position(p, segment)
                    out.append(max_cp)
                    data.append(out)

            # Verify
            for result, max_cp_prior in zip(data, max_cp_priors):
                avg = np.average(result)
                print("prior max_cp={}, p={}, avg_pos={}".format(round(max_cp_prior, 4),
                                                                 round(p, 2),
                                                                 round(avg, 2)))

            self.assertTrue(True)

            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAveragePositionHyperbolicModule))
    unittest.TextTestRunner().run(suite)
