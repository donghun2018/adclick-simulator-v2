# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    from _fix_paths import fix_paths

    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict, List
from abc import abstractmethod
from collections import namedtuple

import ssa_sim_v2.tools.dict_utils as dict_utils

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------


class CompetitiveCPCModule(SimulatorModule):
    """
    Abstract module for generating CPC for segmented data.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary
        that maps segments with functions calculating cpc.
    """
    Params = namedtuple('Params', [])

    def __init__(self, prior={(0,): Params()}, seed=12345):
        """
        :param Dict[tuple, object] prior: Prior.
        :param int seed: Seed for the random number generator.
        """
        super().__init__(prior, seed)

        self.segment_func_map = dict_utils.dict_apply(prior, lambda params: self.generate_cpc_func(params))

    @abstractmethod
    def generate_cpc_func(self, params):
        """
        :param CompetitiveCPCModule params: Params.
        """
        pass

    def get_cpc(self, auction_results, attr=(0,)):
        """
        Returns CPC with optional noise.

        :param Dict[Tuple[int], List[Tuple[int, float]]] auction_results:
            Auction results in the form
            {(0, 0): [(1, bid_11), (2, bid_12), (0, bid_10)],
            (0, 1): [(2, bid_22), (1, bid_21), (0, bid_20)],
            ...}.
        :param tuple attr: Segmentation tuple

        :return: CPC for every ad position.
        :rtype: Union[np.array, list]
        """
        return self.segment_func_map[attr](auction_results[attr])


class CompetitiveCPCVickreyModule(CompetitiveCPCModule):
    """
    Module for calculating prices paid in a Vickrey auction.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dictionary
        that maps segments with functions calculating cpc.
    """

    Params = namedtuple('Params', ['n_pos', 'fee'])

    def generate_cpc_func(self, params):
        """
        :param CompetitiveCPCVickreyModule.Params params: Params.
        The logic here is that:

            1) if the n_pos is less than the length of the auction_results
               we always pay the second price (one at the right) + fee
               for the last element in the array, pay it self + fee
               after the n_pos, we will pay 0
            2) else (if the n_pos is greater than or equal to the length of acution
               we always pay the second price (one at the right) + fee
               for the last element in the array, pay it self + fee
        We will always return the a vector of the cost which will have 
        the same length as the original auction_results. 
        """

        def get_cpc(auction_results):
            len_ = len(auction_results)
            res = []
            n_pos = params.n_pos
            fee = params.fee
            if len_ == 0:
                return []
            if n_pos < len_:
                for i in range(len_):
                    if i == n_pos:
                        break
                    if i < len_ - 1:
                        res.append(auction_results[i + 1][1] + fee)
                    else:
                        res.append(auction_results[-1][1] + fee)
                for i in range(len_ - n_pos):
                    res.append(0.0)
            else:
                for i in range(len_ - 1):
                    res.append(auction_results[i + 1][1] + fee)
                res.append(auction_results[-1][1] + fee)

            return res

        return get_cpc


# ==============================================================================
# Unit tests
# ==============================================================================


if __name__ == "__main__":

    import unittest


    class TestCPCFirstPriceModules(unittest.TestCase):
        def test_sanity(self):
            print("CPCFirstPriceModule class sample run -------------")
            reps = 4
            Params = CompetitiveCPCVickreyModule.Params
            segments = [(seg1, seg2) for seg1 in [0, 1] for seg2 in [0, 1]]
            # bids = np.random.uniform(low=1.0, high=20.0, size=reps)
            # prior = {segment: CPCFirstPriceModule.Params() for segment in segments}
            cpc_model = CompetitiveCPCVickreyModule(
                prior={
                    (0, 0): Params(n_pos=2, fee=0.01),
                    (0, 1): Params(n_pos=3, fee=0.01),
                    (1, 0): Params(n_pos=4, fee=0.01),
                    (1, 1): Params(n_pos=5, fee=0.01)})

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            auction_results = [(1, 2.0), (3, 4.0), (0, 7.0), (4, 8.0), (2, 10.0)]
            auction_results = {(0, 0): auction_results,
                               (0, 1): auction_results,
                               (1, 0): auction_results,
                               (1, 1): auction_results}
            for attr in attributes:
                res = cpc_model.get_cpc(auction_results, attr)
                print("attr={} second price cost ={}".format(attr, res))

            self.assertTrue(True)
            # print out result:
            # attr=(0, 0) second price cost =[4.0, 7.0, 0.0, 0.0, 0.0]
            # attr=(0, 1) second price cost =[4.0, 7.0, 8.0, 0.0, 0.0]
            # attr=(1, 0) second price cost =[4.0, 7.0, 8.0, 10.0, 0.0]
            # attr=(1, 1) second price cost =[4.0, 7.0, 8.0, 10.0, 10.0]

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCPCFirstPriceModules))
    unittest.TextTestRunner().run(suite)
