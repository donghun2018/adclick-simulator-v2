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
from ssa_sim_v2.simulator.modules.revenue.revenue_base_module import RevenueModule

# ------------------------------------------------------------

    
class RevenueDateHoWModule(MultiStateSimulatorModule):
    """
    Module for generating revenue with a separate prior
    for every date-how in a specified range.

    :ivar pd.DataFrame priors: Prior in the form of a DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines class for for given dates and hours of week.
    :ivar Dict[str, RevenueModule] models: Dict with revenue models for every valid pair date and hour of week.
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
        self.models = {}  # type: Dict[str, RevenueModule]

        super().__init__(priors, base_classes, seed)

    def get_revenue(self, num_conversions, date, how, attr=(0,)):
        """
        Returns the revenue based on the base revenue per conversion and the given number of conversions.
        
        :param int num_conversions: Number of conversions.
        :param str date: Date string in the format yyyy-mm-dd.
        :param int how: Integer value for the hour of week in the range 0-167.
        :param tuple attr:

        :return: Total revenue for the given number of conversions.
        :rtype: float
        """
        
        return self.models["{}.{}".format(date, how)].get_revenue(num_conversions, attr)


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":
    
    import unittest
    import math


    class TestRevenueConversionBasedDateHoW(unittest.TestCase):
        def test_sanity(self):
            print("RevenueConversionBasedDateHoW class sample run -------------")

            from ssa_sim_v2.simulator.modules.revenue.revenue_base_module import RevenueModule

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
                if math.floor(float(i) / 12) == 0:
                    Params = RevenueModule.Params
                    priors.loc[index, "prior"] = [{
                        (0, 0): Params(1000.0, noise_level=0.0, noise_type="multiplicative"),
                        (0, 1): Params(1000.0, noise_level=500.0, noise_type="additive"),
                        (1, 0): Params(1000.0, noise_level=0.1, noise_type="multiplicative"),
                        (1, 1): Params(1000.0, noise_level=0.3, noise_type="multiplicative"),
                    }]

                    base_classes.loc[index, "base_class"] = RevenueModule

                else:
                    Params = RevenueModule.Params
                    priors.loc[index, "prior"] = [{
                        (0, 0): Params(3000.0, noise_level=0.0, noise_type="multiplicative"),
                        (0, 1): Params(3000.0, noise_level=500.0, noise_type="additive"),
                        (1, 0): Params(3000.0, noise_level=0.1, noise_type="multiplicative"),
                        (1, 1): Params(3000.0, noise_level=0.3, noise_type="multiplicative"),
                    }]

                    base_classes.loc[index, "base_class"] = RevenueModule

                i = (i + 1) % 24

            model = RevenueDateHoWModule(priors, base_classes, seed=9)

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            conversions_array = np.random.randint(1, 10, 8)
            t = 0

            for attr in attributes:
                while t < len(priors):
                    conversions = conversions_array[t % 8]
                    revenue = model.get_revenue(conversions, priors["date"][t], priors["hour_of_week"][t], attr)
                    print("date={} how={} hod={} attr={} prior_rpv={} conversions={} revenue={} avg_rpv={}".format(
                        priors["date"][t], priors["hour_of_week"][t], priors["hour_of_week"][t] % 24,
                        attr, np.round(priors["prior"][t][attr].avg_rpv, 2), conversions, np.round(revenue, 2),
                        np.round(revenue / conversions, 2)))
                    t += 1
                t = 0

            self.assertTrue(True)

            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRevenueConversionBasedDateHoW))
    unittest.TextTestRunner().run(suite)
