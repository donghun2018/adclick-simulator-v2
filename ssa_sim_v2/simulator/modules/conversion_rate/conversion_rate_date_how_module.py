if __name__ == "__main__":
    from _fix_paths import fix_paths

    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict

import numpy as np
import pandas as pd

from ssa_sim_v2.simulator.modules.multi_state_simulator_module import MultiStateSimulatorModule
from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_base_module import ConversionRateModule

# ------------------------------------------------------------


class ConversionRateDateHoWModule(MultiStateSimulatorModule):
    """
    Module for generating conversion rates with a separate prior
    for every date-how in a specified range.

    :ivar pd.DataFrame priors: Prior in the form of a DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines class for for given dates and hours of week.
    :ivar Dict[str, ConversionRateModule] models: Dict with conversion rate models for every valid pair date and hour of week.
    """

    def __init__(self, priors=None, base_classes=None, seed=12345):
        """
        :param pd.DataFrame priors: DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
        :param pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines class for given dates and hours of week.
        :param int seed: Seed for the random number generator.
        """

        self.models = {}  # type: Dict[str, ConversionRateModule]

        super().__init__(priors, base_classes, seed)

    def get_cvr(self, bid, date, how, attr=(0,)):
        """
        Returns cvr for the given bid using an underlying bid -> average position model
        for the given date and hour of week and attributes.

        :param float bid:
        :param str date: Date string in the format yyyy-mm-dd.
        :param int how: Integer value for the hour of week in the range 0-167.
        :param tuple attr:

        :return: cvr from a bid.
        :rtype: float
        """

        return self.models["{}.{}".format(date, how)].get_cvr(bid, attr)


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":

    import unittest
    import math


    class TestConversionRateDateHoWSegments(unittest.TestCase):
        def test_sanity(self):
            print("ConversionRateDateHoWModule class sample run -------------")

            from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_base_module import \
                ConversionRateFlatModule, ConversionRateFunctionModule

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
                if math.floor(float(i) / 12) == 0:
                    Params = ConversionRateFlatModule.Params
                    priors.loc[index, "prior"] = [{
                        (0, 0): Params(conversion_probability=0.3, noise_level=0.0, noise_type="multiplicative"),
                        (0, 1): Params(conversion_probability=0.5, noise_level=0.1, noise_type="multiplicative"),
                        (1, 0): Params(conversion_probability=0.3, noise_level=0.3, noise_type="multiplicative"),
                        (1, 1): Params(conversion_probability=0.5, noise_level=0.3, noise_type="multiplicative"),
                    }]

                    base_classes.loc[index, "base_class"] = ConversionRateFlatModule

                else:
                    Params = ConversionRateFunctionModule.Params
                    conversion_probabilities = [lambda bid, k=k: min(bid * math.pow(2, 1 / math.sqrt(k + 1)) / 20, 1.0) for k in
                                                range(4)]
                    priors.loc[index, "prior"] = [{
                        (0, 0): Params(conversion_probability=conversion_probabilities[0], noise_level=0.0, noise_type="multiplicative"),
                        (0, 1): Params(conversion_probability=conversion_probabilities[1], noise_level=0.0, noise_type="multiplicative"),
                        (1, 0): Params(conversion_probability=conversion_probabilities[2], noise_level=0.1, noise_type="multiplicative"),
                        (1, 1): Params(conversion_probability=conversion_probabilities[3], noise_level=0.1, noise_type="multiplicative"),
                    }]

                    base_classes.loc[index, "base_class"] = ConversionRateFunctionModule

                i = (i + 1) % 24

            model = ConversionRateDateHoWModule(priors, base_classes, seed=9)

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            bids = np.linspace(1.0, 10.0, 8)
            t = 0

            for attr in attributes:
                while t < len(priors):
                    bid = bids[t % 8]
                    cvr = model.get_cvr(bid, priors["date"][t], priors["hour_of_week"][t], attr)
                    print("date={} how={} hod={} attr={} bid={} cvr={}".format(
                        priors["date"][t], priors["hour_of_week"][t], priors["hour_of_week"][t] % 24,
                        attr, np.round(bid, 2), np.round(cvr, 4)))
                    t += 1
                t = 0

            self.assertTrue(True)

            print("")

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestConversionRateDateHoWSegments))

    unittest.TextTestRunner().run(suite)
