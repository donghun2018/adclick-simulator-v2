# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    from _fix_paths import fix_paths

    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict

import numpy as np
import pandas as pd

from ssa_sim_v2.simulator.modules.multi_state_simulator_module import MultiStateSimulatorModule
from ssa_sim_v2.simulator.modules.auction_attributes.auction_attributes_base_module import AuctionAttributesModule

# ------------------------------------------------------------


class AuctionAttributesDateHoWModule(MultiStateSimulatorModule):
    """
    Module for the determining number of times each segment has been selected using a model specified during initialization.
    For every date and hour of week in a specified range a separate model is used.

    :ivar pd.DataFrame priors: DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar int seed: Seed for the random number generator.
    :ivar Dict[str, AuctionAttributesModule] models: Dictionary of conversions models for every valid pair
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

        self.models = {}  # type: Dict[str, AuctionAttributesModule]

        super().__init__(priors, base_classes, seed)

    def get_auction_attributes(self, n, date="2015-01-05", how=0):
        """
        Returns a dict of number of times each segment has been selected during n auctions using an underlying conversion generation model
        for the given date and hour of week and attributes.

        :param int n: Number of auctions.
        :param str date: Date string in the format yyyy-mm-dd.
        :param int how: Integer value for the hour of week in the range 0-167.

        :return: Dict of number of times each segment was present in n auctions.
        :rtype: Dict[tuple, int]
        """

        return self.models["{}.{}".format(date, how)].get_auction_attributes(n)


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":

    import unittest


    class TestAuctionAttributesDateHoW(unittest.TestCase):
        def test_sanity(self):
            print("AuctionAttributesDateHoWModule class sample run -------------")

            date_range = pd.DataFrame(pd.date_range("2017-01-05", "2017-01-08"), columns=["date"])
            date_range["key"] = 1
            hours = pd.DataFrame(np.array(range(24)), columns=["hour_of_day"])
            hours["key"] = 1
            priors = pd.merge(date_range, hours, how="left", on="key")
            priors["hour_of_week"] = priors["date"].dt.dayofweek * 24 + priors["hour_of_day"]
            priors.loc[:, "date"] = priors["date"].dt.strftime("%Y-%m-%d")
            priors = priors[["date", "hour_of_week"]]

            base_classes = priors.copy()

            priors.loc[:, "prior"] = None
            # print(priors)
            base_classes.loc[:, "base_class"] = None

            i = 0

            for index, row in priors.iterrows():

                Params = AuctionAttributesModule.Params
                priors.loc[index, "prior"] = [{
                    (0, 0): Params(p=45),
                    (0, 1): Params(p=25),
                    (1, 0): Params(p=235),
                    (1, 1): Params(p=76)
                }]

                base_classes.loc[index, "base_class"] = AuctionAttributesModule

                i = (i + 1) % 24

            auctions_attributes_model = AuctionAttributesDateHoWModule(priors, base_classes, seed=9)

            number_of_auctions = [100, 1000, 10000, 15000, 50000, 150000, 300000, 500000]
            t = 0

            while t < len(priors):
                num_auctions = number_of_auctions[t % 8]
                choises_dict = auctions_attributes_model.get_auction_attributes(num_auctions, priors["date"][t], priors["hour_of_week"][t])
                print(
                    "date = {} how = {} number_of_auctions = {} choices = {}".format(priors["date"][t], priors["hour_of_week"][t],
                                                                                     num_auctions, choises_dict))
                t += 1

            self.assertTrue(True)

            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAuctionAttributesDateHoW))
    unittest.TextTestRunner().run(suite)
