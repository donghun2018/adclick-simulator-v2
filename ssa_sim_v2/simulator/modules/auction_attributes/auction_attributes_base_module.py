# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    from _fix_paths import fix_paths

    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict

import numpy as np

from collections import namedtuple

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------


class AuctionAttributesModule(SimulatorModule):
    """
    Base class for all click probability modules with segments.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with constant probabilities for every segment.
    """

    Params = namedtuple('Params', ['p'])
    """
    :param float p: Probability of selecting a user from a segment.
    """

    def __init__(self, prior={(0,): Params(p=5)}, seed=9):
        """

        :param dict prior: Dict with constant probabilities for every segment.
        :param int seed: Seed for the random number generator.
        """
        super().__init__(prior, seed)
        self.prior = dict()
        # Normalize prior and store in self.priors
        total_p_values = 0
        for key in prior.keys():
            total_p_values += prior[key].p
        for key in prior.keys():
            self.prior[key] = AuctionAttributesModule.Params(p=prior[key].p / total_p_values)

        self.rng = np.random.RandomState(seed)

    def get_auction_attributes(self, n):
        """
        Method that returns a dict of number of times each segment has been selected.

        :param int n: Number of auctions for which to sample attributes.
        :return: Dict of number of times each segment was present in n auctions.
        :rtype: Dict[tuple, int]
        """
        # This is used since np does not want to accept tuple as an item and throws error that 'a must be 1-dimensional'
        # dict keys (tuples) are converted to strings, then random choice is made using strings versions of keys, then
        # results are passed to a final dict where keys are of their original form
        keys_dict = dict()
        for key in self.prior.keys():
            keys_dict[str(key)] = key

        keys = list(self.prior)
        keys = [str(key) for key in keys]
        probabilities = [self.prior[keys_dict[key]].p for key in keys]

        choices = self.rng.choice(a=keys, p=probabilities, size=n)
        unique, counts = np.unique(choices, return_counts=True)
        choices_dict_str = dict(zip(unique, counts))
        for key in keys:
            if key in choices_dict_str.keys():
                pass
            else:
                choices_dict_str[key] = 0

        choices_dict = dict()
        for key in self.prior.keys():
            choices_dict[key] = choices_dict_str[str(key)]

        return choices_dict


if __name__ == "__main__":

    import unittest


    class TestAuctionsAttributes(unittest.TestCase):
        def test_sanity(self):

            Params = AuctionAttributesModule.Params

            attributes_model = AuctionAttributesModule(
                prior={
                    (0, 0): Params(p=45),
                    (0, 1): Params(p=25),
                    (1, 0): Params(p=235),
                    (1, 1): Params(p=76)},
                seed=1234
            )

            number_of_auctions = [100, 1000, 10000, 15000, 50000, 150000, 300000, 500000]

            for num in number_of_auctions:
                choices_dict = attributes_model.get_auction_attributes(n=num)
                #print(f'Throughout {num} auctions that were run, following segments were selected following number of times: {choices_dict}')
                print ('Throughout {} auctions that were run, following segments were selected following number of times: {}').format(num, choices_dict)

            self.assertTrue(True)

            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAuctionsAttributes))
    unittest.TextTestRunner().run(suite)
