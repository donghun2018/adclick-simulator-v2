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
from ssa_sim_v2.simulator.modules.conversions.conversions_base_module import ConversionsModule

# ------------------------------------------------------------


class ConversionsDateHoWModule(MultiStateSimulatorModule):
    """
    Module for the determining amount of conversions using a model specified during initialization.
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

        self.models = {}  # type: Dict[str, ConversionsModule]

        super().__init__(priors, base_classes, seed)

    def sample(self, num_clicks, cvr, date="2015-01-05", how=0, attr=(0,)):
        """
        Returns number of conversions for a given number of clicks and conversion rate using an underlying conversion generation model
        for the given date and hour of week and attributes.

        :param int num_clicks: Number of clicks.
        :param float cvr: Conversion rate.
        :param str date: Date string in the format yyyy-mm-dd.
        :param int how: Integer value for the hour of week in the range 0-167.
        :param tuple attr: Attributes.

        :return: Number of conversions for given number of clicks and conversion rate.
        :rtype: float
        """

        return self.models["{}.{}".format(date, how)].sample(num_clicks, cvr, attr)


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":

    import unittest


    class TestConversionsDateHoW(unittest.TestCase):
        def test_sanity(self):
            print("ConversionsDateHoWModule class sample run -------------")

            from ssa_sim_v2.simulator.modules.conversions.conversions_base_module import ConversionsBinomialModule

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

                Params = ConversionsBinomialModule.Params
                priors.loc[index, "prior"] = [{
                    (0, 0): Params(noise_level=0.0, noise_type="multiplicative"),
                    (0, 1): Params(noise_level=0.1, noise_type="multiplicative"),
                    (1, 0): Params(noise_level=0.0, noise_type="additive"),
                    (1, 1): Params(noise_level=0.1, noise_type="additive")
                }]

                base_classes.loc[index, "base_class"] = ConversionsBinomialModule

                i = (i + 1) % 24

            conversions_model = ConversionsDateHoWModule(priors, base_classes, seed=9)

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            num_clicks = np.linspace(1.0, 100.0, 8)
            cvrs = [0.01, 0.02, 0.03, 0.05, 0.10]
            t = 0

            for attr in attributes:
                for cvr in cvrs:
                    while t < len(priors):
                        num_click = num_clicks[t % 8]
                        conversions = conversions_model.sample(num_click, cvr, priors["date"][t], priors["hour_of_week"][t], attr)
                        print(
                            "date={} how={} attr={} num_clicks={} cvr={} conversions={}".format(
                                priors["date"][t], priors["hour_of_week"][t], attr,
                                np.round(num_click, 2), np.round(cvr, 4), conversions))
                        t += 1
                    t = 0

            self.assertTrue(True)

            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestConversionsDateHoW))
    unittest.TextTestRunner().run(suite)
