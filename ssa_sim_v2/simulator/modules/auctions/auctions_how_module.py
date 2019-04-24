# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import numpy as np

from ssa_sim_v2.simulator.modules.auctions.auctions_base import AuctionsPoisson

# ------------------------------------------------------------


class AuctionsPoissonHoW(object):
    """
    action_set module_loader for generating numbers of auctions using Poisson distribution with a separate prior
    for every hour of week.
    
    :ivar Union[list, np.ndarray] L: List of length 168 with average numbers of auctions for every hour of week.
    :ivar list L_len: Length of L.
    :ivar list models: List of 168 auctions models.
    """

    def __init__(self, L=np.random.uniform(size=7*24), seed=9):
        """
        :param Union[list, np.ndarray] L: List of length 168 with average numbers of auctions for every hour of week.
        :param int seed: Seed for the random number generator.
        """
        self.L = L
        self.L_len = len(L)
        
        seed_min = 100000
        seed_max = 999999
        seeds = np.random.RandomState(seed).randint(low=seed_min, high=seed_max, size=168)
        
        self.models = []
        for how in range(self.L_len):
            self.models.append(AuctionsPoisson(L=L[how], seed=seeds[how]))

    def sample(self, how=0):
        """
        Samples the number of auctions.
        
        :param int how: Integer value for the hour of week in the range 0-167.

        :return: action_set randomly chosen number of auctions using Poisson distribution with a mean
            defined for the given hour of week.
        :rtype: int
        """
        
        return self.models[how].sample()


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":
    
    import unittest


    class TestAuctionsPoissonHoW(unittest.TestCase):
        def test_sanity(self):
            print("AuctionsPoissonHoW class sample run -------------")
            L = []
            for how in range(168):
                L.append(np.random.exponential(how+1))
    
            auctions = AuctionsPoissonHoW(L=L)
    
            reps = 10
            data = []
            for rep in range(reps):
                out = []
                for how in range(len(L)):
                    out.append(auctions.sample(how))
                print(out)
                data.append(out)
    
            # Verify
            avg = np.average(data, axis=0)
            for how in range(len(L)):
                print("how={}, l={}, sample mean={}".format(how, L[how], str(avg[how])))
    
            self.assertTrue(True)
            
            print("")
        

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAuctionsPoissonHoW))
    unittest.TextTestRunner().run(suite)
