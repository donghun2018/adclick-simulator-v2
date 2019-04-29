# Fix paths for imports to work in unit tests ----------------


if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from abc import abstractmethod
from collections import namedtuple
import numpy as np

import ssa_sim_v2.tools.dict_utils as dict_utils
from ssa_sim_v2.simulator.modules.commons.noise import NormalNoiseGenerator

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------


class ConversionRateModule(SimulatorModule):
    """
    Abstract module for generating conversion rates for segmented data.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps segments with functions calculating cvr.
    """
    Params = namedtuple('Params', ['cvr', 'noise_level', 'noise_type'])

    def __init__(self, prior={(0,): Params(1.0, 0.0, "multiplicative")}, seed=12345):
        """
        :param dict[tuple, AbstractConversionRateSegments.Params] prior:
        :param int seed: Seed for the random number generator.
        """
        super().__init__(prior, seed)
        self.segment_func_map = dict_utils.dict_apply(prior, lambda params: self.generate_cvr_func(params))

    @abstractmethod
    def generate_cvr_func(self, params):
        """
        :param ConversionRateModule.Params params:
        """
        pass

    def get_cvr(self, bid, attr=(0,)):
        """
        Returns conversion probability with optional noise.

        :param float bid: Base bid value.
        :param tuple attr: Segmentation tuple

        :return: Conversion probability with optional noise.
        :rtype: float
        """
        return self.segment_func_map[attr](bid)


class ConversionRateFlatModule(ConversionRateModule):
    """
    Basic module for generating conversion rates with optional noise for segmented data.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps segments with functions calculating cvr.
    """

    def generate_cvr_func(self, params):
        """
        :param ConversionRateFlatModule.Params params:
        """
        def get_cvr(bid):
            del bid  # Added so that PyCharm doesn't complain
            cvr = NormalNoiseGenerator(params.noise_type).generate_value_with_noise(params.cvr,
                                                                                    params.noise_level,
                                                                                    self.rng)
            return max(min(cvr, 1.0), 0.0)

        return get_cvr


class ConversionRateFunctionModule(ConversionRateModule):
    """
    Basic module for generating conversion rates for segmented data with dependency on bid defined by a function.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps segments with functions calculating cvr.
    """

    def generate_cvr_func(self, params):
        """
        :param ConversionRateFunctionModule.Params params:
        """
        def get_cvr(bid):
            conversion_probability = params.cvr(bid)
            cvr = NormalNoiseGenerator(params.noise_type).generate_value_with_noise(conversion_probability,
                                                                                    params.noise_level,
                                                                                    self.rng)
            return max(min(cvr, 1.0), 0.0)

        return get_cvr


# ==============================================================================
# Unit tests
# ==============================================================================


if __name__ == "__main__":
    
    import unittest
    import math
    import itertools

    class TestConversionRateModule(unittest.TestCase):
        def test_sanity(self):
            print("ConversionRateModule class sample run -------------")
            reps = 100
            bids = np.random.uniform(low=1.0, high=20.0, size=reps)

            segments = [(seg1, seg2) for seg1 in [0, 1] for seg2 in [0, 1]]
            cvrs = np.random.uniform(low=0.1, high=0.5, size=len(segments))
            noise_levels = [0, 0.1, 0.1, 0.3]
            noise_types = ["multiplicative", "multiplicative", "additive", "multiplicative", ]
            prior = {segment: ConversionRateModule.Params(cvr, noise_level, noise_type) for
                     segment, cvr, noise_level, noise_type in
                     zip(segments, cvrs, noise_levels, noise_types)}

            res = []
            conversions_model = ConversionRateFlatModule(prior, seed=np.random.randint(1, 100000))
            for attr, bid in itertools.product(segments, bids):
                cvr = conversions_model.get_cvr(bid, attr)
                res.append(cvr)
                print("segments={}, cvr={}, bid={}, cvr={}".format(
                    attr, np.round(prior[attr].cvr, 4), np.round(bid, 2), round(cvr, 4)))


    class TestConversionRateFunctionModule(unittest.TestCase):
        def test_sanity(self):
            print("ConversionRateFunctionModule class sample run -------------")
            reps = 100

            segments = [(seg1, seg2) for seg1 in [0, 1] for seg2 in [0, 1]]
            cvrs = [lambda bid, k=k: min(bid * math.pow(2, 1 / math.sqrt(k + 1)) / 20, 1.0)
                    for k in range(len(segments))]
            noise_levels = [0, 0.1, 0.1, 0.3]
            noise_types = ["multiplicative", "multiplicative", "additive", "multiplicative", ]
            prior = {segment: ConversionRateModule.Params(cvr, noise_level, noise_type) for
                     segment, cvr, noise_level, noise_type in
                     zip(segments, cvrs, noise_levels, noise_types)}

            bids = np.random.uniform(low=0.0, high=20.0, size=reps)

            res = []
            conversions_model = ConversionRateFunctionModule(prior, seed=np.random.randint(1, 100000))
            for attr, bid in itertools.product(segments, bids):
                cvr = conversions_model.get_cvr(bid, attr)
                res.append(cvr)
                print("segments={}, conversion_probability={}, bid={}, cvr={}".format(
                    attr, np.round(prior[attr].cvr(bid), 4), np.round(bid, 2), np.round(cvr, 4)))


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestConversionRateModule))
    suite.addTest(unittest.makeSuite(TestConversionRateFunctionModule))

    unittest.TextTestRunner().run(suite)
