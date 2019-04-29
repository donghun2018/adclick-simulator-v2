# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    from _fix_paths import fix_paths

    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from abc import abstractmethod
from typing import Dict, List
import numpy as np
from collections import namedtuple

import ssa_sim_v2.tools.dict_utils as dict_utils
from ssa_sim_v2.simulator.modules.commons.noise import NormalNoiseGenerator

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------

class CompetitiveClicksModule(SimulatorModule):
    """
    Basic module for generating numbers of clicks using binomial distribution.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps segments with functions calculating number of obtained clicks.
    """

    Params = namedtuple('Params', ['noise_level', 'noise_type'])
    """
    :param float noise_level: Noise level.
    :param float noise_type: Noise type:

        * additive -- normal noise with noise_level as the standard deviation,
        * multiplicative -- normal noise with noise_level as the standard deviation times the base value.
    """

    def __init__(self, prior={(0,): Params(noise_level=0.0, noise_type="multiplicative")}, seed=12345):
        """
        :param Dict[tuple, ClicksBinomialSegments.Params] prior: dictionary segments as keys to which a configuration of approriate noise_level and noise_type is assigned
        :param int seed: Seed for the random number generator.
        """
        super().__init__(prior, seed)

        self.segment_func_map = dict_utils.dict_apply(prior, self.generate_clicks_func)

    @abstractmethod
    def generate_clicks_func(self, params):
        """
        :param ClicksModule.Params params:
        """
        pass

    def sample(self, num_auctions, cp, attr):
        """
        Samples the number of clicks using binomial distribution from the given number of auctions
        and a click probability.

        :param int num_auctions: Number of auctions
        :param List[np.array, list] cp: Array of click probabilities.
        :param attr: segment for which the computation is done, e.g. (1,0)
        :return: Array of number of clicks for every ad position.
        :rtype: Union[np.array, list]
        """

        return self.segment_func_map[attr](num_auctions, cp)


class CompetitiveClicksBinomialModule(CompetitiveClicksModule):
    """
    Basic module for generating numbers of clicks using binomial distribution.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps segments with functions calculating number of obtained clicks.
    """

    Params = namedtuple('Params', ['noise_level', 'noise_type'])
    """
    :param float noise_level: Noise level.
    :param float noise_type: Noise type:

        * additive -- normal noise with noise_level as the standard deviation,
        * multiplicative -- normal noise with noise_level as the standard deviation times the base value.
    """

    def generate_clicks_func(self, params):
        """
        :param CompetitiveClicksBinomialModule.Params params:
        """
        noise_type = params.noise_type
        noise_level = params.noise_level
        assert (noise_type == "additive" or noise_type == "multiplicative")
        # Note the cp must be in vector form
        # clip each value in the cp and then take the binomial distribution vector
        def sample(num_auctions, cp):
            cp_vec = []
            for prob in cp:
                p = NormalNoiseGenerator(noise_type).generate_value_with_noise(
                    prob, noise_level, self.rng)
                p = np.clip(p, 0.0, 1.0)
                cp_vec.append(p)
            return (self.rng.binomial(num_auctions, cp_vec))

        return sample


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":

    import unittest

    class TestClicksBinomialModule(unittest.TestCase):
        def test_sanity(self):
            reps = 1000

            Params = CompetitiveClicksBinomialModule.Params

            clicks_model = CompetitiveClicksBinomialModule(
                prior={
                    (0, 0): Params(noise_level=0.1, noise_type="multiplicative"),
                    (0, 1): Params(noise_level=0.2, noise_type="multiplicative"),
                    (1, 0): Params(noise_level=0.3, noise_type="additive"),
                    (1, 1): Params(noise_level=0.4, noise_type="additive")})

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            # replace the test with a vector of cp
            for attr in attributes:
                vec_cp = [0.15, 0.25, 0.35]
                sample_sum = np.zeros(len(vec_cp))
                for r in range(reps):
                    sample = clicks_model.sample(np.random.uniform(low=0.0, high=100.0), [0.15, 0.25, 0.35], attr)
                    sample_sum += sample
                print("attr={} average_clicks={}".format(attr,
                                                         np.round(sample_sum / reps, 2)))

            self.assertTrue(True)

            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestClicksBinomialModule))

    unittest.TextTestRunner().run(suite)
