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
from ssa_sim_v2.simulator.modules.average_position.average_position_base_module import AveragePositionModule

# ------------------------------------------------------------


class AveragePositionDateHoWModule(MultiStateSimulatorModule):
    """
    Module for the average position in AdWords results for segmented data based on the max click probability
    and the actual probability with a separate prior for every date and hour of week in a specified range.
    
    A normal noise can be added to this value.
    
    :ivar pd.DataFrame priors: Prior in the form of a DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines class for for given dates and hours of week.
    :ivar Dict[str, AveragePositionModule] models: Dict with average price models for every valid pair date and hour of week.
    """

    def __init__(self, priors=None, base_classes=None, seed=123):
        """
        :param pd.DataFrame priors: DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
        :param pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines class for given dates and hours of week.
        :param int seed: Seed for the random number generator.
        """

        self.models = {}  # type: Dict[str, AveragePositionModule]

        super().__init__(priors, base_classes, seed)

    def get_average_position(self, click_probability, date, how, attr=(0,)):
        """
        Returns average position for the given click probability using an underlying click probability -> average
        position model for the given date and hour of week and attributes.

        :param float click_probability:
        :param str date: Date string in the format yyyy-mm-dd.
        :param int how: Integer value for the hour of week in the range 0-167.
        :param tuple attr:

        :return: Average position from a click probability.
        :rtype: float
        """

        return self.models["{}.{}".format(date, how)].get_average_position(click_probability, attr)


# ==============================================================================
# Unit tests
# ==============================================================================


if __name__ == "__main__":

    import unittest
    from ssa_sim_v2.simulator.modules.average_position.average_position_base_module import AveragePositionHyperbolicModule


    class TestAveragePositionHyperbolicDateHoWModule(unittest.TestCase):
        def test_sanity(self):
            print("AveragePositionDateHoWModule class sample run -------------")

            import math

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
                if math.floor(float(i) / 8) == 0:
                    Params = AveragePositionHyperbolicModule.Params
                    priors.loc[index, "prior"] = [{
                        (0, 0): Params(max_cp=1.0, noise_level=0.1, noise_type="multiplicative"),
                        (0, 1): Params(max_cp=0.5, noise_level=0.1, noise_type="multiplicative"),
                        (1, 0): Params(max_cp=0.3, noise_level=0.1, noise_type="multiplicative"),
                        (1, 1): Params(max_cp=0.1, noise_level=0.1, noise_type="multiplicative"),
                    }]

                    base_classes.loc[index, "base_class"] = AveragePositionHyperbolicModule

                elif math.floor(float(i) / 8) == 1:
                    Params = AveragePositionHyperbolicModule.Params
                    priors.loc[index, "prior"] = [{
                        (0, 0): Params(max_cp=1.0, noise_level=0.1, noise_type="additive"),
                        (0, 1): Params(max_cp=0.5, noise_level=0.1, noise_type="additive"),
                        (1, 0): Params(max_cp=0.3, noise_level=0.1, noise_type="additive"),
                        (1, 1): Params(max_cp=0.1, noise_level=0.1, noise_type="additive"),
                    }]

                    base_classes.loc[index, "base_class"] = AveragePositionHyperbolicModule

                elif math.floor(float(i) / 8) == 2:
                    Params = AveragePositionHyperbolicModule.Params
                    priors.loc[index, "prior"] = [{
                        (0, 0): Params(max_cp=1.0, noise_level=0.5, noise_type="multiplicative"),
                        (0, 1): Params(max_cp=0.5, noise_level=0.5, noise_type="multiplicative"),
                        (1, 0): Params(max_cp=0.3, noise_level=0.5, noise_type="multiplicative"),
                        (1, 1): Params(max_cp=0.1, noise_level=0.5, noise_type="multiplicative"),
                    }]

                    base_classes.loc[index, "base_class"] = AveragePositionHyperbolicModule

                i = (i + 1) % 24

            model = AveragePositionDateHoWModule(priors, base_classes, seed=9)

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            cps = np.linspace(0.1, 1.0, 8)
            t = 0

            for attr in attributes:
                while t < len(priors):
                    cp = cps[t % 8]
                    avg_pos = model.get_average_position(cp, priors["date"][t], priors["hour_of_week"][t], attr)
                    print("date={} how={} hod={} attr={} cp={} avg_pos={}".format(
                        priors["date"][t], priors["hour_of_week"][t],
                        priors["hour_of_week"][t] % 24, attr, round(cp, 2), avg_pos))
                    t += 1
                t = 0

            self.assertTrue(True)

            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAveragePositionHyperbolicDateHoWModule))
    unittest.TextTestRunner().run(suite)
