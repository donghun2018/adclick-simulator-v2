# Fix paths for imports to work in unit tests ----------------
from typing import Dict

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import numpy as np
import pandas as pd

from ssa_sim_v2.simulator.modules.auctions.auctions_base import AuctionsPoisson

# ------------------------------------------------------------


class AuctionsPoissonDate(object):
    """
    action_set module_loader for generating numbers of auctions using Poisson distribution with a separate prior
    for every date in a specified range.
    
    :ivar pd.DataFrame L: Prior in the form of a DataFrame with two columns: date, auctions.
        The last column defines an average numbers of auctions for a given date.
    :ivar int L_len: Length of L.
    :ivar dict models: Dictionary of auctions models for every valid pair date.
    """

    def __init__(self, L=None, seed=9):
        """
        :param pd.DataFrame L:  DataFrame with three columns: date, auctions. The last column defines an average
            numbers of auctions for a given date.
        :param int seed: Seed for the random number generator.
        """
        
        if L is None:
            # Define a default_cvr prior spanning 2015 and 2016
            default_prior = pd.DataFrame(pd.date_range("2015-01-05", "2017-01-01"), columns=["date"])
            default_prior.loc[:, "date"] = default_prior["date"].dt.strftime("%Y-%m-%d")
            default_prior["auctions"] = np.random.randint(low=0, high=100, size=len(default_prior))
            default_prior = default_prior[["date", "auctions"]]
            
            L = default_prior
        
        self.L = L
        self.L_len = len(L)
        
        seed_min = 100000
        seed_max = 999999
        seeds = np.random.RandomState(seed).randint(low=seed_min, high=seed_max, size=self.L_len)
        
        self.models = {}
        for t in range(self.L_len):
            self.models[self.L["date"][t]] = AuctionsPoisson(L=self.L["auctions"][t], seed=seeds[t])

    def sample(self, date="2015-01-05"):
        """
        Samples the number of auctions.

        :param str date: Date string in the format yyyy-mm-dd.

        :return: action_set randomly chosen number of auctions using Poisson distribution with a mean
            defined for the given date.
        :rtype: int
        """

        return self.models[date].sample()
    

# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":
    
    import unittest


    class TestAuctionsPoissonDate(unittest.TestCase):
        def test_sanity(self):
            print("AuctionsPoissonDate class sample run -------------")
            
            default_prior = pd.DataFrame(pd.date_range("2015-01-05", "2017-01-01"), columns=["date"])
            default_prior.loc[:, "date"] = default_prior["date"].dt.strftime("%Y-%m-%d")
            default_prior["auctions"] = np.random.randint(low=0, high=100, size=len(default_prior))
            default_prior = default_prior[["date", "auctions"]]
    
            auctions = AuctionsPoissonDate(default_prior, seed=9)
    
            reps = 10
            data = []
            for rep in range(reps):
                out = []
                for t in range(len(default_prior)):
                    out.append(auctions.sample(default_prior["date"][t]))
                print(out)
                data.append(out)
    
            # Verify
            avg = np.average(data, axis=0)
            for t in range(len(default_prior)):
                print("date={}, l={}, sample mean={}".format(default_prior["date"][t], 
                      default_prior["auctions"][t], avg[t]))
    
            self.assertTrue(True)
            
            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAuctionsPoissonDate))
    unittest.TextTestRunner().run(suite)
