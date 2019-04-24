# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import numpy as np

from collections import namedtuple

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------


class AuctionsModule(SimulatorModule):
    """
    Basic module for generating numbers of auctions using Poisson distribution.

    :ivar dict prior: Dict with priors.
    :ivar np.random.RandomState rng: Random number generator.
    """

    Params = namedtuple('Params', ['auctions'])
    """
    :param float auctions: Average number of auctions.
    """

    def __init__(self, prior={(): Params(auctions=100)}, seed=9):
        """
        :param dict prior: Dict with priors.
        :param int seed: Seed for the random number generator.
        """

        super().__init__(prior, seed)

    def sample(self):
        """Samples the number of auctions.

        :return: action_set randomly chosen number of auctions using Poisson distribution with mean L.
        :rtype: int
        """

        return self.rng.poisson(self.prior[()].auctions)


class AuctionsPoissonModule(AuctionsModule):
    """
    Basic module for generating numbers of auctions using Poisson distribution.
    
    :ivar dict prior: Dict with priors.
    :ivar np.random.RandomState rng: Random number generator.
    """

    Params = namedtuple('Params', ['auctions'])
    """
    :param float auctions: Average number of auctions.
    """


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":
    
    import unittest

    class TestAuctionsPoisson(unittest.TestCase):
        def test_sanity(self):
            print("AuctionsPoisson class sample run -------------")
            lambdas = np.random.exponential(10, size=10)
            reps = 100
    
            for l in lambdas:
                out = []
                auctions = AuctionsPoissonModule({(): AuctionsPoissonModule.Params(auctions=l)})
                for r in range(reps):
                    out.append(auctions.sample())
    
                avg = np.average(out)
                print("L={} sample mean={}".format(l, avg))
    
            self.assertTrue(True)
            
            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAuctionsPoisson))
    unittest.TextTestRunner().run(suite)
