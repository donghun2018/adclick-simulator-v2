# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict
from collections import namedtuple
import numpy as np

from ssa_sim_v2.simulator.modules.commons.noise import NormalNoiseGenerator
import ssa_sim_v2.tools.dict_utils as dict_utils

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------

# TODO: Check docstrings
# TODO: Fix the unit test, add test for the gamma noise module
# TODO: Make sure that there are no warnings given by PyCharm left
# TODO: Delete these TODOs


class RevenueModule(SimulatorModule):
    """
    Base module for the revenue based on the average conversion value (revenue) and the number of conversions.
    
    :ivar object rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps segments with functions calculating cvr.
    """
    Params = namedtuple('Params', ['avg_rpv', 'noise_level', 'noise_type'])
    """
    :param float avg_rpv: Historical average value per conversion.
    :param float noise_level: Noise level.
    :param str noise_type: Noise type:

        * additive -- normal noise with noise_level as the standard deviation,
        * multiplicative -- normal noise with noise_level as the standard deviation times the base value
            for the revenue per conversion.
    """

    def __init__(self, prior={(0,): Params(1.0, 0.0, "multiplicative")}, seed=12345):
        """
        :param Dict[tuple, AbstractConversionRateSegments.Params] prior:
        :param int seed: Seed for the random number generator.
        """
        super().__init__(prior, seed)

        self.segment_func_map = dict_utils.dict_apply(prior, lambda params: self.generate_revenue_func(params))

        self.last_rpv = 0.0

    def generate_revenue_func(self, params):
        """
        :param RevenueModule.Params params:
        """
        def get_revenue(num_conversions):
            rpv = NormalNoiseGenerator(params.noise_type).generate_value_with_noise(
                params.avg_rpv, params.noise_level, self.rng)
            self.last_rpv = np.maximum(0.0, rpv)

            return self.last_rpv * num_conversions

        return get_revenue

    def get_revenue(self, num_conversions, attr=(0,)):
        """
        Returns the revenue for segmented data based on the base revenue per conversion and the given number of conversions.

        :param int num_conversions: Number of conversions.
        :param tuple attr: Segmentation tuple

        :return: Total revenue for the given number of conversions.
        :rtype: float
        """
        return self.segment_func_map[attr](num_conversions)


class RevenueNormalNoiseModule(RevenueModule):
    """
    Module for the revenue based on the average conversion value (revenue) and the number of conversions.

    :ivar object rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps segments with functions calculating cvr.
    """
    Params = namedtuple('Params', ['avg_rpv', 'noise_level', 'noise_type'])
    """
    :param float avg_rpv: Historical average value per conversion.
    :param float noise_level: Noise level.
    :param str noise_type: Noise type:

        * additive -- normal noise with noise_level as the standard deviation,
        * multiplicative -- normal noise with noise_level as the standard deviation times the base value
            for the revenue per conversion.
    """

    def generate_revenue_func(self, params):
        """
        :param RevenueModule.Params params:
        """
        def get_revenue(num_conversions):
            rpv = NormalNoiseGenerator(params.noise_type).generate_value_with_noise(
                params.avg_rpv, params.noise_level, self.rng)
            self.last_rpv = np.maximum(0.0, rpv)

            return self.last_rpv * num_conversions

        return get_revenue


class RevenueGammaNoiseModule(RevenueModule):
    """
    Basic module_loader for the revenue based on the average conversion value (revenue) and the number of conversions.

    :ivar object rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary that maps segments with functions calculating cvr.
    """

    Params = namedtuple('Params', ['avg_rpv', 'noise_level'])
    """
    :param float avg_rpv: Historical average value per conversion. It is the guaranteed mean for revenue samples.
    :param float noise_level: Noise level > 0. Shape parameter of gamma. Larger values generate heavier tails.

    """

    def generate_revenue_func(self, params):
        """
        params.avg_rpv is the guaranteed mean for this revenue
        :param RevenueModule.Params params:
        """
        # TODO: Add gamma noise as a new Noise class and us it here.
        def get_revenue(num_conversions):

            sampling_num = 1 if num_conversions == 0 else num_conversions

            rpv = self.rng.gamma(shape=params.noise_level,
                                 scale=params.avg_rpv/params.noise_level,
                                 size=sampling_num)

            self.last_rpv = np.mean(rpv)

            return self.last_rpv * num_conversions

        return get_revenue


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":
    
    import unittest
    import itertools

    class TestRevenueConversionBasedModule(unittest.TestCase):
        def test_sanity(self):
            print("RevenueModule class sample run -------------")
            reps = 25

            segments = [(seg1, seg2) for seg1 in [0, 1] for seg2 in [0, 1]]
            rpvs = np.random.uniform(low=1000.0, high=4000.0, size=len(segments))
            noise_levels = [0, 0.1, 0.1, 0.3]
            noise_types = ["multiplicative", "multiplicative", "additive", "multiplicative", ]
            prior = {segment: RevenueModule.Params(rpv, noise_level, noise_type) for
                     segment, rpv, noise_level, noise_type in
                     zip(segments, rpvs, noise_levels, noise_types)}

            num_conversions = np.random.poisson(0.2, size=reps)
            res = []
            revenue_model = RevenueModule(prior)
            for attr, n in itertools.product(segments, num_conversions):
                rev = revenue_model.get_revenue(n, attr)
                res.append(rev)
                print("segments={}, rpv={}, n={}, revenue={}".format(
                    attr, np.round(prior[attr].avg_rpv, 2), n, np.round(rev, 2)))
                
            self.assertTrue(True)


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRevenueConversionBasedModule))
    unittest.TextTestRunner().run(suite)
