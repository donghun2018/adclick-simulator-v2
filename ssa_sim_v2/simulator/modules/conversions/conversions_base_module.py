# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    from _fix_paths import fix_paths

    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from abc import abstractmethod
from typing import Dict
import numpy as np
import ssa_sim_v2.tools.dict_utils as dict_utils
from collections import namedtuple

from ssa_sim_v2.simulator.modules.commons.noise import NormalNoiseGenerator

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------


class ConversionsModule(SimulatorModule):
    """
    Basic module for generating numbers of conversions using binomial distribution with a given probability.

    A normal noise can be added to this value.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps
        segments with functions calculating average_position.
    """

    Params = namedtuple('Params', ['noise_level', 'noise_type'])
    """
    :param float noise_level: Noise level.
    :param float noise_type: Noise type:

        * additive -- normal noise with noise_level as the standard deviation,
        * multiplicative -- normal noise with noise_level as the standard deviation times the base value.
    """

    def __init__(self, prior={(0,): Params(noise_level=0.0, noise_type= "multiplicative")}, seed=12345):
        """
        :param Dict[tuple, ConversionsModule.Params] prior: dictionary segments
            as keys to which a configuration of appropriate noise_level and noise_type is assigned.
        :param int seed: Seed for the random number generator.
        """
        super().__init__(prior, seed)

        self.segment_func_map = dict_utils.dict_apply(prior, self.generate_conversions_func)

    @abstractmethod
    def generate_conversions_func(self, params):
        """
        :param ConversionsModule.Params params:
        """
        pass

    def sample(self, num_clicks, cvr, attr):
        """
        Samples the number of conversions using binomial distribution from the given number of clicks

        :param num_clicks: int num_clicks: Number of clicks (trials).
        :param cvr: Probability -- conversion rate.
        :param attr: segment for which the computation is done, e.g. (1,0)
        :return: A randomly chosen number of conversions.
        :rtype: int
        """

        return self.segment_func_map[attr](num_clicks, cvr)


class ConversionsBinomialModule(ConversionsModule):
    """
    Basic module for generating numbers of conversions using binomial distribution with a given probability.

    A normal noise can be added to this value.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps
     segments with functions calculating average_position.
    """

    Params = namedtuple('Params', ['noise_level', 'noise_type'])
    """
    :param float noise_level: Noise level.
    :param float noise_type: Noise type:

        * additive -- normal noise with noise_level as the standard deviation,
        * multiplicative -- normal noise with noise_level as the standard deviation times the base value.
    """

    def generate_conversions_func(self, params):
        """
        :param ConversionsBinomialModule.Params params:
        """
        noise_type = params.noise_type
        noise_level = params.noise_level
        assert (noise_type == "additive" or noise_type == "multiplicative")

        def get_conversions(n, cvr):
            cvr = NormalNoiseGenerator(noise_type).generate_value_with_noise(
                cvr, noise_level, self.rng)

            p = np.clip(cvr, 0.0, 1.0)

            return self.rng.binomial(n, p)

        return get_conversions


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":

    import unittest

    class TestConversionsBinomialModule(unittest.TestCase):
        def test_sanity(self):
            print("ConversionsBinomialModule class sample run -------------")
            reps = 1000

            Params = ConversionsBinomialModule.Params

            conversions_model = ConversionsBinomialModule(
                prior={
                    (0, 0): Params(noise_level=0.1, noise_type="multiplicative"),
                    (0, 1): Params(noise_level=0.2, noise_type="multiplicative"),
                    (1, 0): Params(noise_level=0.3, noise_type="additive"),
                    (1, 1): Params(noise_level=0.4, noise_type="additive")})

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]

            for attr in attributes:
                sample_sum = 0
                for r in range(reps):
                    sample = conversions_model.sample(np.random.uniform(low=0.0, high=100.0), 0.15, attr)
                    sample_sum += sample
                print("attr = {} avg_conversions = {}".format(attr, round(sample_sum/reps, 2)))

            self.assertTrue(True)

            print("")

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestConversionsBinomialModule))

    unittest.TextTestRunner().run(suite)
