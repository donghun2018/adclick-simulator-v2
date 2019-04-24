# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict, List

import numpy as np
import pandas as pd

from ssa_sim_v2.simulator.modules.multi_state_simulator_module import MultiStateSimulatorModule
from ssa_sim_v2.simulator.modules.competitive_cpc.competitive_cpc_base_module import CompetitiveCPCModule

# ------------------------------------------------------------

# TODO: Check docstrings
# TODO: Fix the unit test
# TODO: Make sure that there are no warnings given by PyCharm left
# TODO: Delete these TODOs

    
class CompetitiveCpcDateHoWModule(MultiStateSimulatorModule):
    """
    Module for the bid->cpc relation using a model specified during initialization.
    For every date and hour of week in a specified range a separate model is used.
    
    :ivar pd.DataFrame priors: DataFrame with three columns: date, hour_of_week,
            prior. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar pd.DataFrame base_classes: DataFrame with three columns: date, hour_of_week,
            base_class. The last column defines priors (in the form of dictionaries)
            for given dates and hours of week.
    :ivar int seed: Seed for the random number generator.
    :ivar Dict[str, ClickProbabilityModule]: Dictionary of click probability models for every valid pair
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

        self.models = {}  # type: Dict[str, CompetitiveCPCModule]

        super().__init__(priors, base_classes, seed)

    def get_cpc(self, auction_results, date="2015-01-05", how=0, attr=(0,)):
        """
        Returns a cost per click for all ad positions using an underlying cpc model
        for the given date and hour of week and attributes.
        
        :param Dict[Tuple[int], List[Tuple[int, float]]] auction_results:
            Auction results in the form
            {(0, 0): [(1, bid_11), (2, bid_12), (0, bid_10)],
            (0, 1): [(2, bid_22), (1, bid_21), (0, bid_20)],
            ...}.
        :param str date: Date string in the format yyyy-mm-dd.
        :param int how: Integer value for the hour of week in the range 0-167.
        :param tuple attr: Attributes.

        :return: Array of costs per click for every ad position.
        :rtype: List[int]
        """
        
        return self.models["{}.{}".format(date, how)].get_cpc(auction_results, attr)


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":
    
    import unittest
            
            
    class TestClickProbabilityDateHoW(unittest.TestCase):
        def test_sanity(self):
            print("ClickProbabilityDateHoWModule class sample run -------------")

            import math

            from ssa_sim_v2.simulator.modules.click_probability.click_probability_base_module import ClickProbabilityFunctionModule
            from ssa_sim_v2.simulator.modules.click_probability.click_probability_base_module import ClickProbabilityLogisticLogS1Module
            from ssa_sim_v2.simulator.modules.click_probability.click_probability_base_module import ClickProbabilityLogisticLogShiftModule
            from ssa_sim_v2.simulator.modules.click_probability.click_probability_date_how_module import \
                ClickProbabilityDateHoWModule

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
                    Params = ClickProbabilityFunctionModule.Params
                    priors.loc[index, "prior"] = [{
                        (0, 0): Params(bid=1.0, p=lambda b: min(1.0, max(0.0, b / 10))),
                        (0, 1): Params(bid=1.0, p=lambda b: min(0.5, max(0.0, b / 10))),
                        (1, 0): Params(bid=1.0, p=lambda b: min(1.0, max(0.0, b / 10)) / 2),
                        (1, 1): Params(bid=1.0, p=lambda b: min(0.5, max(0.0, b / 10)) / 2)
                    }]

                    base_classes.loc[index, "base_class"] = ClickProbabilityFunctionModule

                elif math.floor(float(i) / 8) == 1:
                    Params = ClickProbabilityLogisticLogS1Module.Params
                    priors.loc[index, "prior"] = [{
                        (0, 0): Params(bid=5.0, p=0.1),
                        (0, 1): Params(bid=5.0, p=0.3),
                        (1, 0): Params(bid=5.0, p=0.5),
                        (1, 1): Params(bid=5.0, p=0.7)
                    }]

                    base_classes.loc[index, "base_class"] = ClickProbabilityLogisticLogS1Module

                elif math.floor(float(i) / 8) == 2:
                    Params = ClickProbabilityLogisticLogShiftModule.Params
                    priors.loc[index, "prior"] = [{
                        (0, 0): Params(bid=[], p=[], theta_0=1.0, theta_1=0.5, max_cp=0.5, tau=3.0, fit_type="lr"),
                        (0, 1): Params(bid=[], p=[], theta_0=1.0, theta_1=0.5, max_cp=0.5, tau=6.0, fit_type="lr"),
                        (1, 0): Params(bid=[], p=[], theta_0=1.0, theta_1=0.5, max_cp=0.5, tau=9.0, fit_type="lr"),
                        (1, 1): Params(bid=[], p=[], theta_0=1.0, theta_1=0.5, max_cp=0.5, tau=12.0, fit_type="lr")
                    }]

                    base_classes.loc[index, "base_class"] = ClickProbabilityLogisticLogShiftModule

                i = (i + 1) % 24

            click_prob_model = ClickProbabilityDateHoWModule(priors, base_classes, seed=9)

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            bids = np.linspace(1.0, 20.0, 8)
            t = 0

            for attr in attributes:
                while t < len(priors):
                    bid = bids[t % 8]
                    cp = click_prob_model.get_cp(bid, priors["date"][t], priors["hour_of_week"][t], attr)
                    print("date={} how={} attr={} bid={} cp={}".format(priors["date"][t], priors["hour_of_week"][t], attr, bid, cp))
                    t += 1
                t = 0

            self.assertTrue(True)
            
            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestClickProbabilityDateHoW))
    unittest.TextTestRunner().run(suite)
