# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict
import numpy as np
from abc import abstractmethod
from collections import namedtuple, defaultdict

from ssa_sim_v2.simulator.modules.commons.noise import NormalNoiseGenerator
import ssa_sim_v2.tools.dict_utils as dict_utils

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------


class CPCModule(SimulatorModule):
    """
    Abstract module for generating CPC for segmented data.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary
        that maps segments with functions calculating cpc.
    """
    Params = namedtuple('Params', [])

    def __init__(self, prior={(0,): Params()}, seed=12345):
        """
        :param Dict[tuple, object] prior:
        :param int seed: Seed for the random number generator.
        """
        super().__init__(prior, seed)

        self.segment_func_map = dict_utils.dict_apply(prior, lambda params: self.generate_cpc_func(params))

    @abstractmethod
    def generate_cpc_func(self, params):
        """
        :param object params:
        """
        pass

    def get_cpc(self, bid, attr=(0,)):
        """
        Returns CPC with optional noise.

        :param float bid: Base bid value.
        :param tuple attr: Segmentation tuple

        :return: CPC with optional noise.
        :rtype: float
        """
        return self.segment_func_map[attr](bid)


class CPCFirstPriceModule(CPCModule):
    """
    Basic module for the cost per click assuming that always the bid value is paid.
    It can be thought of as a first price auction - everybody places a bid, gets his position 
    according to a ranking of bids and pays his bid for this position.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary
        that maps segments with functions calculating cpc.
    """

    def generate_cpc_func(self, params):
        """
        :param CPCModule.Params params:
        """
        def get_cpc(bid):
            return bid

        return get_cpc

    
class CPCSimpleSecondPriceModule(CPCModule):
    """
    Basic module for the cost per click assuming that there are two competing players in an auction,
    both with the same expected value of their belief for the real value they compete for. Then we assume 
    that the information of the second player is imperfect, adding a normal noise to his choice of a bid. 
    Moreover, we assume that a losing player pays his bid. This leads to the noise model given by a capped 
    normal distribution.
    
    :ivar np.random.RandomState rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary
        that maps segments with functions calculating cpc.
    """
    Params = namedtuple('Params', ['noise_level', 'noise_type'])

    def generate_cpc_func(self, params):
        """
        :param CPCSimpleSecondPriceModule.Params params:
        """
        def get_cpc(bid):
            cpc = NormalNoiseGenerator(params.noise_type).generate_value_with_noise(bid, params.noise_level, self.rng)
            return np.clip(cpc, 0.0, bid)

        return get_cpc


class CPCBidHistoricalAvgCPCModule(CPCModule):
    """
    Basic module for the cost per click returning the bid value capped at the historical average cpc.
    It can be thought as a second price auction with only one competitor with his bid equal to the 
    historical average cpc.
    
    action_set normal noise can be added to this value (positive noise is reduced to zero). Rationale for the choice
    of this probability distribution: we assume two competing players in an auction, both with the same
    expected value of their belief for the real value they compete for. Then we assume that the information 
    of the second player is imperfect, adding a normal noise to his choice of a bid. Moreover, we assume that 
    a losing player pays his bid. This leads to the noise model given by a capped normal distribution.
    
    :ivar np.random.RandomState rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary
        that maps segments with functions calculating cpc.
    """

    Params = namedtuple('Params', ['avg_hist_cpc', 'noise_level', 'noise_type'])

    def generate_cpc_func(self, params):
        """
        :param CPCBidHistoricalAvgCPCModule.Params params:
        """
        def get_cpc(bid):
            cpc = NormalNoiseGenerator(params.noise_type).generate_value_with_noise(
                params.avg_hist_cpc, params.noise_level, self.rng)
            return np.clip(cpc, 0.0, bid)

        return get_cpc


class CPCBidMinusCpcDiffModule(CPCModule):
    """
    Basic module for the cost per click returning the bid value minus the average historical difference
    between bids and actual costs per click. Symbolically,
    cpc = bid - avg([hist_bid] - [hist_cpc])
    
    action_set normal noise can be added to this value. Rationale for the choice of this probability distribution: we
    assume two competing players in an auction, the second with his belief equal to the belief of the first 
    minus the historical difference. Then we assume that the information of the second player is imperfect, 
    adding a normal noise to his choice of a bid.
    
    :ivar np.random.RandomState rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary
        that maps segments with functions calculating cpc.
    """

    Params = namedtuple('Params', ['avg_hist_bid', 'avg_hist_cpc', 'noise_level', 'noise_type'])
    """
    :param float avg_hist_bid: Historical average bid.
    :param float avg_hist_cpc: Historical average cost per click.
    :param float noise_level: Noise level.
    :param str noise_type: Noise type:
        * additive -- normal noise with noise_level as the standard deviation,
        * multiplicative -- normal noise with noise_level as the standard deviation times the base value
            for the cpc.
    """

    def generate_cpc_func(self, params):
        """
        :param CPCBidMinusCpcDiffModule.Params params:
        """
        avg_cpc_diff = np.maximum(params.avg_hist_bid - params.avg_hist_cpc, 0.0)

        def get_cpc(bid):
            cpc = bid - avg_cpc_diff
            cpc = NormalNoiseGenerator(params.noise_type).generate_value_with_noise(
                cpc, params.noise_level, self.rng)
            return np.clip(cpc, 0.0, bid)

        return get_cpc

# ==============================================================================
# Unit tests
# ==============================================================================


if __name__ == "__main__":

    import unittest
    import itertools

    class TestCPCFirstPriceModules(unittest.TestCase):
        def test_sanity(self):
            print("CPCFirstPriceModule class sample run -------------")
            reps = 100
            bids = np.random.uniform(low=1.0, high=20.0, size=reps)

            segments = [(seg1, seg2) for seg1 in [0, 1] for seg2 in [0, 1]]
            prior = {segment: CPCFirstPriceModule.Params() for segment in segments}

            res = []
            cpc_model = CPCFirstPriceModule(prior, seed=np.random.randint(1, 100000))
            for attr, bid in itertools.product(segments, bids):
                cpc = cpc_model.get_cpc(bid, attr)
                self.assertEqual(bid, cpc)
                res.append(cpc)
                print("segments={}, bid={}, cpc={}".format(attr, bid, round(cpc, 4)))


    class TestCPCSimpleSecondPriceModules(unittest.TestCase):
        def test_sanity(self):
            print("CPCSimpleSecondPriceModule class sample run -------------")
            reps = 100
            bids = np.random.uniform(low=1.0, high=20.0, size=reps)

            Params = CPCSimpleSecondPriceModule.Params

            segments = [(seg1, seg2) for seg1 in [0, 1] for seg2 in [0, 1]]
            noise_levels = [0.0, 0.1, 0.1, 0.9]
            noise_types = ["multiplicative", "multiplicative", "additive", "multiplicative", ]
            prior = {segment: Params(noise_level, noise_type) for
                     segment, noise_level, noise_type in
                     zip(segments, noise_levels, noise_types)}

            res = defaultdict(list)
            cpc_model = CPCSimpleSecondPriceModule(prior, seed=np.random.randint(1, 100000))
            for attr, bid in itertools.product(segments, bids):
                cpc = cpc_model.get_cpc(bid, attr)
                res[attr].append(cpc)
                print("segments={}, bid={}, cpc={}".format(attr, bid, np.round(cpc, 4)))

            for attr, cpcs in res.items():
                cpcs_diff = bids - np.array(cpcs)
                expected_lower_bound = bids - bids * prior[attr].noise_level
                print(attr, expected_lower_bound.sum(), np.array(cpcs).sum())
                self.assertTrue(expected_lower_bound.sum() <= np.array(cpcs).sum())
                self.assertTrue((cpcs_diff >= 0).all())

    class TestCPCBidHistoricalAvgCPCModules(unittest.TestCase):
        def test_sanity(self):
            print("CPCSimpleSecondPriceModule class sample run -------------")
            reps = 100
            bids = np.random.uniform(low=1.0, high=20.0, size=reps)

            Params = CPCBidHistoricalAvgCPCModule.Params

            avg_cpcs = [5.0, 10.0, 15.0]
            segments = [(seg1, seg2, seg3) for seg1 in [0, 1] for seg2 in [0, 1] for seg3 in range(len(avg_cpcs))]
            noise_levels = [0, 0.1, 0.1, 0.9] * len(avg_cpcs)
            noise_types = ["multiplicative", "multiplicative", "additive", "multiplicative", ] * len(avg_cpcs)
            prior = {segment: Params(avg_hist_cpc=avg_cpc, noise_level=noise_level, noise_type=noise_type) for
                     segment, avg_cpc, noise_level, noise_type in
                     zip(segments, avg_cpcs*4, noise_levels, noise_types)}

            res = defaultdict(list)
            cpc_model = CPCBidHistoricalAvgCPCModule(prior, seed=np.random.randint(1, 100000))
            for attr, bid in itertools.product(segments, bids):
                cpc = cpc_model.get_cpc(bid, attr)
                res[attr].append(cpc)
                print("segments={}, bid={}, cpc={}".format(attr, bid, np.round(cpc, 4)))

            for attr, cpcs in res.items():
                cpcs_diff = bids - np.array(cpcs)

                self.assertTrue((cpcs_diff >= 0).all())

                avg_cpc = prior[attr].avg_hist_cpc
                expected_lower_bound = bids.clip(0, avg_cpc) * (1 - prior[attr].noise_level)
                print(expected_lower_bound.mean(), np.array(cpcs).mean())
                self.assertTrue(expected_lower_bound.mean() <= np.array(cpcs).mean())

                expected_upper_bound = avg_cpc + avg_cpc * prior[attr].noise_level
                self.assertTrue(expected_upper_bound >= np.array(cpcs).mean())


    class TestCPCBidMinusCpcDiffModules(unittest.TestCase):
        def test_sanity(self):
            print("CPCBidMinusCpcDiffSegmentsclass sample run -------------")
            reps = 5
            bids = np.random.uniform(low=1.0, high=20.0, size=reps)

            Params = CPCBidMinusCpcDiffModule.Params

            avg_cpcs = [5.0, 10.0, 15.0]
            avg_hist_bids = [6.0, 10.0, 20.0]
            segments = [(seg1, seg2, seg3) for seg1 in [0, 1] for seg2 in [0, 1] for seg3 in range(len(avg_cpcs))]
            noise_levels = [0, 0.1, 0.1, 0.9] * len(avg_cpcs)
            noise_types = ["multiplicative", "multiplicative", "additive", "multiplicative", ] * len(avg_cpcs)
            prior = {segment: Params(avg_bid, avg_cpc, noise_level, noise_type) for
                     segment, avg_bid, avg_cpc, noise_level, noise_type in
                     zip(segments, avg_hist_bids*4, avg_cpcs*4, noise_levels, noise_types)}

            res = defaultdict(list)
            cpc_model = CPCBidMinusCpcDiffModule(prior, seed=np.random.randint(1, 100000))
            for attr, bid in itertools.product(segments, bids):
                cpc = cpc_model.get_cpc(bid, attr)
                res[attr].append(cpc)
                print("segments={}, avg_hist_bid={}, avg_hist_cpc={}, bid={}, cpc={}".format(
                    attr, np.round(prior[attr].avg_hist_bid, 2),
                    np.round(prior[attr].avg_hist_cpc, 2), np.round(bid, 2), np.round(cpc, 2)))

            for attr, cpcs in res.items():
                cpc_diff = prior[attr].avg_hist_bid - prior[attr].avg_hist_cpc

                self.assertTrue(cpc_diff >= 0)

                expected_lower_bound = (bids - cpc_diff) * (1 - prior[attr].noise_level)
                print(expected_lower_bound.mean(), np.array(cpcs).mean())
                self.assertTrue(expected_lower_bound.mean() <= np.array(cpcs).mean())

                expected_upper_bound = (bids - cpc_diff).clip(0) * (1 + prior[attr].noise_level)
                print(expected_upper_bound.mean(), np.array(cpcs).mean())
                self.assertTrue(expected_upper_bound.mean() >= np.array(cpcs).mean())

                self.assertTrue(bids.mean() >= np.array(cpcs).mean())


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCPCFirstPriceModules))
    suite.addTest(unittest.makeSuite(TestCPCSimpleSecondPriceModules))
    suite.addTest(unittest.makeSuite(TestCPCBidHistoricalAvgCPCModules))
    suite.addTest(unittest.makeSuite(TestCPCBidMinusCpcDiffModules))

    unittest.TextTestRunner().run(suite)
