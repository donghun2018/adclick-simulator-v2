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

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------

# TODO: Check docstrings
# TODO: Fix the unit test
# TODO: Make sure that there are no warnings given by PyCharm left
# TODO: Delete these TODOs


class CompetitiveClickProbabilityModule(SimulatorModule):
    """
    Base class for all competitive click probability modules with segments.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with dummy bids and constant probabilities for every segment.
    :ivar Dict[tuple, Callable[[float], float]] segment_func_map: Dict with functions
        returning constant probability for every bid for every segment.
    """

    Params = namedtuple('Params', [])

    def __init__(self, prior={(0,): Params()}, seed=9):
        """
        :param dict prior: Dict with dummy bids and constant probabilities for every segment.
        :param int seed: Seed for the random number generator.
        """

        super().__init__(prior, seed)

        self.segment_func_map = dict_utils.dict_apply(prior, self.generate_click_probability_func)

    @abstractmethod
    def generate_click_probability_func(self, params):
        """
        :param CompetitiveClickProbabilityModule.Params params: Params.
        """
        pass

    def get_cp(self, auction_results, attr=(0,)):
        """
        Returns an array of click probabilities for the given auction results
        using an underlying bid->click probability model.

        :param Dict[Tuple[int], List[Tuple[int, float]]] auction_results:
            Auction results in the form
            {(0, 0): [(1, bid_11), (2, bid_12), (0, bid_10)],
            (0, 1): [(2, bid_22), (1, bid_21), (0, bid_20)],
            ...}.
        :param tuple attr: Attributes.

        :return: Array of click probabilities for every ad position.
        :rtype: Union[np.array, list]
        """

        return self.segment_func_map[attr](auction_results[attr])


class CompetitiveClickProbabilityTwoClassGeometricModule(CompetitiveClickProbabilityModule):
    """
    Module for the click probability assuming three classes of users:

        1) Those who do not click on ads.
        2) Those who aim to find one relevant ad.
        3) Those who aim to compare ads.
    The percentage of classes 2 and 3 is given by parameter p. The percentage
    of people in class 2 is given by p * q, while of people in class 3 by p * (1 - q).
    It is assumed that the probability of a person from class 2 clicking on the
    first ad is r_11, and then geometrically drops with the coefficient r_12.
    This is to signify the importance of the first position. For a person from
    class 2 there is an r_2 chance of this person clicking only on the first ad,
    and then geometrically decreases for the chance of clicking on the first two ads,
    first three ads and so on.
    
    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with dummy bids and constant probabilities for every segment.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dict with functions
        returning constant probability for every bid for every segment.
    """

    Params = namedtuple('Params', ['n_pos', 'p', 'q', 'r_11', 'r_12', 'r_2'])
    """
    :param int n_pos: Number of available ad positions.
    :param float p: Percentage of people who actually want two click.
    :param float q: Percentage of people who choose the first relevant ad and only click on it. 
        This class of people is called I. Then 1 - q is the percentage of people who compare offers.
        The second class of people is called II.
    :param float r_11: Geometric distribution probability of people from the first class
        clicking on the first position. Then all other cases have probability 1 - r_11.
    :param float r_12: Geometric distribution probability that a person from the first class
        who did not click on the first ad will click on the second ad. Then a click
        on the third ad has a probability of (1 - r_11) (1 - r_12) * r_12, on the fourth
        has probability (1 - r_11) (1 - r_12)^2 * r_12 and so on. The first position has
        its separate probability to signify the importance of being on the first spot. 
    :param float r_2: Geometric distribution probability for the second class of people.
        The first probability gives the percentage of people who click only on the first
        position. The second probability gives the percentage of people who click
        on the first two ads. The third probability is for the first three ads and so on.
    """

    def generate_click_probability_func(self, params):
        """
        :param CompetitiveClickProbabilityTwoClassGeometricModule.Params params: Params.
        """

        n_pos = params.n_pos
        p = params.p
        q = params.q
        r_11 = params.r_11
        r_12 = params.r_12
        r_2 = params.r_2

        # TODO: Code this method
        def get_cp(auction_results):
            prob_1 = np.array([r_11 if i == 0 else 0.0 for i in range(n_pos)]) \
                  + (1 - r_11) * r_12 * np.array([(1 - r_12) ** (k - 1) if k > 0 else 0.0 for k in range(n_pos)])
            prob_1 = prob_1 / np.sum(prob_1) * p * q
            prob_2 = r_2 * np.array([(1 - r_2) ** k for k in range(n_pos)])
            prob_2 = prob_2 / np.sum(prob_2) * p * (1 - q)
            prob_2 = np.cumsum(prob_2[::-1])[::-1]

            return np.array([prob_1[i] + prob_2[i] if i < n_pos else 0.0 for i in range(len(auction_results))])

        return get_cp


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":
    
    import unittest


    class TestCompetitiveClickProbabilityTwoClassGeometricModule(unittest.TestCase):
    
        def test_sanity(self):
            print("CompetitiveClickProbabilityTwoClassGeometricModule class sample run -------------")

            Params = CompetitiveClickProbabilityTwoClassGeometricModule.Params

            prior = {
                    (0, 0): Params(n_pos=10, p=0.31, q=0.5, r_11=0.3, r_12=0.2, r_2=0.15),
                    (0, 1): Params(n_pos=10, p=0.33, q=0.5, r_11=0.3, r_12=0.2, r_2=0.15),
                    (1, 0): Params(n_pos=10, p=0.28, q=0.5, r_11=0.3, r_12=0.2, r_2=0.15),
                    (1, 1): Params(n_pos=10, p=0.35, q=0.5, r_11=0.3, r_12=0.2, r_2=0.15)
            }
    
            click_prob_model = CompetitiveClickProbabilityTwoClassGeometricModule(
                prior=prior
            )

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]

            for attr in attributes:
                p = click_prob_model.generate_click_probability_func(prior[attr])
                p = p(0.1)
                print("attr={} p={}".format(attr, p))
    
            self.assertTrue(True)
            
            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCompetitiveClickProbabilityTwoClassGeometricModule))
    unittest.TextTestRunner().run(suite)
