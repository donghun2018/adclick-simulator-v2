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
from ssa_sim_v2.simulator.modules.clicks.clicks_base_module import ClicksModule

# ------------------------------------------------------------


class ClicksDateHoWModule(MultiStateSimulatorModule):
    """
    Module for the determining amount of clicks using a model specified during initialization.
    For every date and hour of week in a specified range a separate model is used.

    :ivar pd.DataFrame priors: DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar int seed: Seed for the random number generator.
    :ivar Dict[str, ClicksModule] models: Dictionary of conversions models for every valid pair
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

        self.models = {}  # type: Dict[str, ClicksModule]

        super().__init__(priors, base_classes, seed)

    def sample(self, num_auctions, cp, date="2015-01-05", how=0, attr=(0,)):
        """
        Returns number of clicks for a given number of auctions and click probability using an underlying clicks generating model
        for a given date and hour of week and attributes.

        :param int num_auctions: Number of clicks.
        :param float cp: Click probability.
        :param str date: Date string in the format yyyy-mm-dd.
        :param int how: Integer value for the hour of week in the range 0-167.
        :param tuple attr: Attributes determining segment.

        :return: Number of clicks for given number of auctions and click probability.
        :rtype: float
        """

        return self.models["{}.{}".format(date, how)].sample(num_auctions, cp, attr)


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":

    import unittest


    class TestClicksDateHoW(unittest.TestCase):
        def test_sanity(self):
            print("ClicksDateHoWModule class sample run -------------")

            from ssa_sim_v2.simulator.modules.clicks.clicks_base_module import ClicksBinomialModule

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
            base_classes.loc[:, "base_class"] = None

            i = 0

            for index, row in priors.iterrows():

                Params = ClicksBinomialModule.Params
                priors.loc[index, "prior"] = [{
                    (0, 0): Params(noise_level=0.0, noise_type="multiplicative"),
                    (0, 1): Params(noise_level=0.1, noise_type="multiplicative"),
                    (1, 0): Params(noise_level=0.0, noise_type="additive"),
                    (1, 1): Params(noise_level=0.1, noise_type="additive")
                }]

                base_classes.loc[index, "base_class"] = ClicksBinomialModule

                i = (i + 1) % 24

            clicks_model = ClicksDateHoWModule(priors, base_classes, seed=9)

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            num_auctions = np.linspace(10.0, 100.0, 8)
            cps = [0.01, 0.02, 0.03, 0.05, 0.10]
            t = 0

            for attr in attributes:
                for cp in cps:
                    while t < len(priors):
                        num_auction = num_auctions[t % 8]
                        clicks = clicks_model.sample(num_auction, cp, priors["date"][t], priors["hour_of_week"][t], attr)
                        print(
                            "date={} how={} attr={} num_auctions={} cp={} clicks={}".format(
                                priors["date"][t], priors["hour_of_week"][t], attr,
                                round(num_auction, 2), cp, clicks))
                        t += 1
                    t = 0

            self.assertTrue(True)

            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestClicksDateHoW))
    unittest.TextTestRunner().run(suite)
