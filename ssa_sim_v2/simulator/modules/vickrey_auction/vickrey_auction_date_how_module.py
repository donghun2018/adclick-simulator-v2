# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    from _fix_paths import fix_paths

    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict

import numpy as np
import pandas as pd

from ssa_sim_v2.simulator.action import Action
from ssa_sim_v2.simulator.modules.multi_state_simulator_module import MultiStateSimulatorModule
from ssa_sim_v2.simulator.modules.vickrey_auction.vickrey_auction_module import VickreyAuctionModule

# ------------------------------------------------------------


class VickreyAuctionDateHoWModule(MultiStateSimulatorModule):
    """
    Module for performing Vickrey auctions using a model specified during initialization.
    For every date and hour of week in a specified range a separate model is used.

    :ivar pd.DataFrame priors: DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar int seed: Seed for the random number generator.
    :ivar Dict[str, ConversionsModule] models: Dictionary of conversions models for every valid pair
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

        self.models = {}  # type: Dict[str, VickreyAuctionModule]

        super().__init__(priors, base_classes, seed)

    def get_auction_results(self, actions, date="2015-01-05", how=0):
        """
        Returns results of vickrey auction using an underlying vickrey auction model with attributes stored in the model
        for the given date and hour of week.

        :param list actions: list of objects of type Action(bid, modifiers) which are basically decisions
         of policies that have entered the auction
        :param str date: Date string in the format yyyy-mm-dd.
        :param int how: Integer value for the hour of week in the range 0-167.

        :return: results of auction for each attribute combination(segment).
        :rtype: dict
        """

        return self.models["{}.{}".format(date, how)].get_auction_results(actions)


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":

    import unittest


    class TestVickreyAuctionDateHoW(unittest.TestCase):
        def test_sanity(self):
            print("VickreyAuctionDateHoWModule class sample run -------------")

            from ssa_sim_v2.simulator.modules.vickrey_auction.vickrey_auction_module import VickreyAuctionModule

            date_range = pd.DataFrame(pd.date_range("2017-01-05", "2017-01-06"), columns=["date"])
            date_range["key"] = 1
            hours = pd.DataFrame(np.array(range(24)), columns=["hour_of_day"])
            hours["key"] = 1
            priors = pd.merge(date_range, hours, how="left", on="key")
            priors["hour_of_week"] = priors["date"].dt.dayofweek * 24 + priors["hour_of_day"]
            priors.loc[:, "date"] = priors["date"].dt.strftime("%Y-%m-%d")
            priors = priors[["date", "hour_of_week"]]

            base_classes = priors.copy()

            priors.loc[:, "prior"] = None
            base_classes.loc[:, "base_class"] = None

            i = 0

            for index, row in priors.iterrows():

                Params = VickreyAuctionModule.Params
                priors.loc[index, "prior"] = [{
                    (0, 0): Params(),
                    (0, 1): Params(),
                    (1, 0): Params(),
                    (1, 1): Params()
                }]

                base_classes.loc[index, "base_class"] = VickreyAuctionModule

                i = (i + 1) % 24

            vickrey_model = VickreyAuctionDateHoWModule(priors, base_classes, seed=9)

            actions = [Action(2.0, [[np.random.uniform(0.5, 1.5) for _ in range(2)]
                                    for _ in range(2)]) for _ in range(3)]
            t = 0

            while t < len(priors):
                results = vickrey_model.get_auction_results(actions=actions, date=priors["date"][t], how=priors["hour_of_week"][t])
                print("date={} how={} result={}".format(priors["date"][t], priors["hour_of_week"][t], results))
                t = t + 1

            self.assertTrue(True)

            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestVickreyAuctionDateHoW))
    unittest.TextTestRunner().run(suite)
