# Fix paths for imports to work in unit tests ----------------
from typing import Dict

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import numpy as np
import pandas as pd

from ssa_sim_v2.simulator.modules.multi_state_simulator_module import MultiStateSimulatorModule
from ssa_sim_v2.simulator.modules.auctions.auctions_base_module import AuctionsModule

# ------------------------------------------------------------


class AuctionsDateHoWModule(MultiStateSimulatorModule):
    """
    A module_loader for generating numbers of auctions using Poisson distribution with a separate prior
    for every date and hour of week in a specified range.

    :ivar pd.DataFrame priors: DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar int seed: Seed for the random number generator.
    :ivar Dict[str, AuctionsModule] models: Dictionary of conversions models for every valid pair
        of date and hour of week.
    """

    def __init__(self, priors=None, base_classes=None, seed=9):
        """
        :param pd.DataFrame priors: DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
        :param pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
        :param int seed: Seed for the random number generator.
        """
        self.models = {}  # type: Dict[str, AuctionsModule]

        super().__init__(priors, base_classes, seed)

    def sample(self, date="2015-01-05", how=0):
        """
        Samples the number of auctions.

        :param str date: Date string in the format yyyy-mm-dd.
        :param int how: Integer value for the hour of week in the range 0-167.

        :return: action_set randomly chosen number of auctions using Poisson distribution with a mean
            defined for the given date and hour of week.
        :rtype: int
        """

        return self.models["{}.{}".format(date, how)].sample()


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":
    
    import unittest


    class TestAuctionsPoissonDateHoW(unittest.TestCase):
        def test_sanity(self):
            print("AuctionsPoissonDateHoW class sample run -------------")

            from ssa_sim_v2.simulator.modules.auctions.auctions_base_module import AuctionsPoissonModule
            
            date_range = pd.DataFrame(pd.date_range("2015-01-05", "2017-01-01"), columns=["date"])
            date_range["key"] = 1
            hours = pd.DataFrame(np.array(range(24)), columns=["hour_of_day"])
            hours["key"] = 1
            priors = pd.merge(date_range, hours, how="left", on="key")
            priors["hour_of_week"] = priors["date"].dt.dayofweek * 24 + priors["hour_of_day"]
            priors.loc[:, "date"] = priors["date"].dt.strftime("%Y-%m-%d")
            priors = priors[["date", "hour_of_week"]]

            base_classes = priors.copy()

            priors.loc[:, "prior"] = [{(): AuctionsPoissonModule.Params(auctions=l)}
                                      for l in np.random.exponential(10, size=len(priors))]
            base_classes.loc[:, "base_class"] = AuctionsPoissonModule
    
            auctions = AuctionsDateHoWModule(priors=priors, base_classes=base_classes, seed=9)
    
            data = []
            out = []
            
            for t in range(len(priors)):
                out.append(auctions.sample(priors["date"][t], priors["hour_of_week"][t]))

                if priors["hour_of_week"][t] == 167:
                    data.append(out)
                    out = []
    
            # Calculate avg priors
            
            avg_priors = np.zeros(168)
            
            for how in range(168):
                avg_priors[how] = auctions.priors.loc[auctions.priors["hour_of_week"] == how, "prior"].apply(lambda x: x[()].auctions).mean()
    
            # Verify
            avg = np.average(data, axis=0)
            for how in range(168):
                print("how={}, how mean={}, sample mean={}".format(
                    how, round(avg_priors[how], 2), round(avg[how], 2)))
    
            self.assertTrue(True)
            
            print("")
        

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAuctionsPoissonDateHoW))
    unittest.TextTestRunner().run(suite)