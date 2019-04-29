# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from abc import abstractmethod
from collections import namedtuple

import numpy as np

import ssa_sim_v2.tools.dict_utils as dict_utils

from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------


class ClickProbabilityModule(SimulatorModule):
    """
    Base class for all click probability modules with segments.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with dummy bids and constant probabilities for every segment.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dict with functions
        returning constant probability for every bid for every segment.
    """

    Params = namedtuple('Params', ['bid', 'p'])
    """
    :param float bid: Bid.
    :param float p: Click probability.
    """

    def __init__(self, prior={(0,): Params(bid=1.0, p=0.5)}, seed=9):
        """
        :param dict prior: Dict with dummy bids and constant probabilities for every segment.
        :param int seed: Seed for the random number generator.
        """

        super().__init__(prior, seed)

        self.segment_func_map = dict_utils.dict_apply(prior, self.generate_click_probability_func)

    def get_cp(self, bid=np.random.uniform(low=0, high=20), attr=(0,)):
        """
        Returns a click probability for the given bid using an underlying bid->click probability model.

        :param float bid: Bid value.
        :param tuple attr: Attributes.

        :return: Click probability for the given bid.
        :rtype: float
        """

        return self.segment_func_map[attr](bid)

    @abstractmethod
    def generate_click_probability_func(self, params):
        """
        :param ClickProbabilityModule.Params params: Params.
        """
        pass


class ClickProbabilityConstantModule(ClickProbabilityModule):
    """
    Basic module for the bid->click probability relation always returning a chosen constant probability.
    
    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with dummy bids and constant probabilities for every segment.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dict with functions
        returning constant probability for every bid for every segment.
    """

    Params = namedtuple('Params', ['bid', 'p'])
    """
    :param float bid: Bid.
    :param float p: Click probability.
    """

    def generate_click_probability_func(self, params):
        """
        :param ClickProbabilityConstantModule.Params params: Params.
        """
        return lambda x: params.p


class ClickProbabilityFunctionModule(ClickProbabilityModule):
    """
    Basic module for the bid->click probability relation using a predefined function.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with dummy bids and a probability function for every segment.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dict with functions returning a probability for every bid for every segment.
    """

    Params = namedtuple('Params', ['bid', 'p'])
    """
    :param float bid: Bid.
    :param function p: Click probability function.
    """

    def generate_click_probability_func(self, params):
        """
        :param ClickProbabilityFunctionModule.Params params: Params.
        """
        return params.p
    
    
class ClickProbabilityLogisticS1Module(ClickProbabilityModule):
    """
    Basic module for the bid->click probability relation using a logistic curve with slope equal to 1.
    A logistic curve going through a predefined point (bid, click probability) is used for every segment.
    
    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with bid and click probability for every segment.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dict with functions returning a probability for every bid for every segment.
    """

    Params = namedtuple('Params', ['bid', 'p'])
    """
    :param float bid: Bid.
    :param float p: Click probability.
    """

    def generate_click_probability_func(self, params):
        """
        :param ClickProbabilityLogisticS1Module.Params params: Params.
        """
        bid = params.bid
        p = params.p
        if p < 0.001:
            p = 0.01
        if p > 0.999:
            p = 0.999
        theta_1 = 1.0
        theta_0 = np.log(p / (1 - p)) - theta_1 * bid
        return lambda b: 1 / (1 + np.exp(-theta_0 - theta_1 * b)) if b != 0 else 0


class ClickProbabilityLogisticLogS1Module(ClickProbabilityModule):
    """
    Basic module for the bid->click probability relation using a logistic curve with slope equal to 1
    composed with the natural logarithm. The composition with logarithm ensures click probability to be zero
    for bid equal to zero. A curve going through a predefined point (bid, click probability) is used
    for every segment.
    
    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with bid and click probability for every segment.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dict with functions returning
        a probability for every bid for every segment.
    """

    Params = namedtuple('Params', ['bid', 'p'])
    """
    :param float bid: Bid.
    :param float p: Click probability.
    """

    def generate_click_probability_func(self, params):
        """
        :param ClickProbabilityLogisticLogS1Module.Params params: Params.
        """
        bid = params.bid
        p = params.p
        if p < 0.001:
            p = 0.01
        if p > 0.999:
            p = 0.999
        theta_1 = 1.0
        theta_0 = np.log(p / (1 - p)) - theta_1 * np.log(bid)
        return lambda b: 1 / (1 + np.exp(-theta_0 - theta_1 * np.log(b))) if b != 0 else 0
    

class ClickProbabilityLogisticModule(ClickProbabilityModule):
    """
    Basic module modeling the bid->click probability relation using a logistic curve
    for every segment. Three parameters are used to describe a logistic curve:
        * theta_0 -- the "intercept",
        * theta_1 -- the "slope" (thought as the coefficients in the exponent),
        * max_cp - the upper asymptote.

    The following formula is used:

    .. math::
        \\frac{max\_cp}{1 + e^{-theta_0 - theta_1 bid}}

    Any subset of those parameters can be preset in the initialization. If left
    as None, max_cp is taken to be twice the historical maximal click
    probability and theta_0 and theta_1 are found using the least squares method.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with bids, click probabilities and/or some logistic curve params for every segment:

        * list bid -- List of bid values.
        * list bid -- List of click probabilities.
        * float theta_0 -- "Intercept" for the logistic curve.
        * float theta_1 -- "Slope" for the logistic curve.
        * float max_cp -- Upper asymptote for the logistic curve.
        * str fit_type -- Type of fitting to be applied:

            * "lr" -- Linear regression fitting only theta_0 and theta_1,
                while max_cp is estimated from the historical maximum.
            * "gd" -- Gradient descent to find the optimal theta_0, theta_1
                and max_cp.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dict with functions returning
        a probability for every bid for every segment.
    """

    Params = namedtuple('Params', ['bid', 'p', 'theta_0', 'theta_1', 'max_cp', 'fit_type'])
    """
    :param list bid: List of bid values
    :param list bid: List of click probabilities.
    :param float theta_0: "Intercept" for the logistic curve.
    :param float theta_1: "Slope" for the logistic curve.
    :param float max_cp: Upper asymptote for the logistic curve.
    :param str fit_type: Type of fitting to be applied:

        * "lr" -- Linear regression fitting only theta_0 and theta_1,
            while max_cp is estimated from the historical maximum.
        * "gd" -- Gradient descent to find the optimal theta_0, theta_1
            and max_cp.
    """

    def generate_click_probability_func(self, params):
        """
        :param ClickProbabilityLogisticModule.Params params: Params.
        """
        bid = params.bid
        p = params.p
        theta_0 = params.theta_0
        theta_1 = params.theta_1
        max_cp = params.max_cp
        fit_type = params.fit_type

        assert(len(p) == len(bid))

        # Make p and bid a numpy array and artificially add data points to make sure prob(0.0) = 0.0
        p = np.append(np.array(p), [0.0001])  # option: use len(p) or max(int(math.sqrt(len(p))), 1) zeros
        bid = np.append(np.array(bid), [0.0001])

        p = np.minimum(np.maximum(p, 0.0001), 0.9999)

        if fit_type == "lr":

            n = len(p)

            # Set click probability max

            if max_cp is None:
                max_cp = min(1.5 * max(p), 1.0)

            # Calculate missing coefficients using the least squares method

            if theta_0 is not None and theta_1 is None:

                theta_1 = (sum(bid * np.log(p / (max_cp - p))) - theta_0 * sum(bid)) / sum(bid * bid)

            elif theta_0 is None and theta_1 is not None:

                theta_0 = (sum(np.log(p / (max_cp - p))) - theta_1 * sum(bid)) / n

            elif theta_0 is None and theta_1 is None:

                theta_1 = (n * sum(bid * np.log(p / (max_cp - p))) - sum(bid) * sum(np.log(p / (max_cp - p)))) \
                    / (n * sum(bid * bid) - sum(bid)**2)
                theta_0 = (sum(np.log(p / (max_cp - p))) - theta_1 * sum(bid)) / n

            # Define the probability curve

            return lambda b: max_cp / (1 + np.exp(-theta_0 - theta_1 * b)) if b != 0 else 0


class ClickProbabilityLogisticLogModule(ClickProbabilityModule):
    """
    Basic module modeling the bid->click probability relation using a logistic curve composed with
    the natural logarithm for every segment. Three parameters are used to describe this curve:
        * theta_0 -- the "intercept",
        * theta_1 -- the "slope" (thought as the coefficients in the exponent),
        * max_cp - the upper asymptote.

    The following formula is used:

    .. math::
        \\frac{max\_cp}{1 + e^{-theta_0 - theta_1 \\ln(bid)}}

    Any subset of those parameters can be preset in the initialization. If left as None, max_cp is taken 
    to be twice the historical maximal click probability and theta_0 and theta_1 are found using 
    the least squares method.
    
    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with bids, click probabilities and/or some logistic curve params for every segment:

        * list bid -- List of bid values.
        * list bid -- List of click probabilities.
        * float theta_0 -- "Intercept" for the logistic curve.
        * float theta_1 -- "Slope" for the logistic curve.
        * float max_cp -- Upper asymptote for the logistic curve.
        * str fit_type -- Type of fitting to be applied:

            * "lr" -- Linear regression fitting only theta_0 and theta_1,
                while max_cp is estimated from the historical maximum.
            * "gd" -- Gradient descent to find the optimal theta_0, theta_1
                and max_cp.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dict with functions returning
        a probability for every bid for every segment.
    """

    Params = namedtuple('Params', ['bid', 'p', 'theta_0', 'theta_1', 'max_cp', 'fit_type'])
    """
    :param list bid: List of bid values
    :param list bid: List of click probabilities.
    :param float theta_0: "Intercept" for the logistic curve.
    :param float theta_1: "Slope" for the logistic curve.
    :param float max_cp: Upper asymptote for the logistic curve.
    :param str fit_type: Type of fitting to be applied:

        * "lr" -- Linear regression fitting only theta_0 and theta_1,
            while max_cp is estimated from the historical maximum.
        * "gd" -- Gradient descent to find the optimal theta_0, theta_1
            and max_cp.
    """

    def generate_click_probability_func(self, params):
        """
        :param ClickProbabilityLogisticLogModule.Params params: Params.
        """
        bid = params.bid
        p = params.p
        theta_0 = params.theta_0
        theta_1 = params.theta_1
        max_cp = params.max_cp
        fit_type = params.fit_type

        assert(len(p) == len(bid))

        # If there is only one distinct data point, add an artificial data point near (0, 0)
        if len(set(bid)) == 1:
            p = np.append(np.array(p), [0.0001])  # option: use len(p) or max(int(math.sqrt(len(p))), 1) zeros
            bid = np.append(np.array(bid), [0.0001])
        else:
            p = np.array(p)
            bid = np.array(bid)

        p = np.minimum(np.maximum(p, 0.0001), 0.9999)

        if fit_type == "lr":

            n = len(p)

            # Set click probability max

            if max_cp is None:
                max_cp = min(1.5 * max(p), 1.0)

            # Calculate missing coefficients using the least squares method

            if theta_0 is not None and theta_1 is None:

                theta_1 = (sum(np.log(bid) * np.log(p / (max_cp - p))) - theta_0 * sum(np.log(bid))) \
                    / sum(np.log(bid) * np.log(bid))

            elif theta_0 is None and theta_1 is not None:

                theta_0 = (sum(np.log(p / (max_cp - p))) - theta_1 * sum(np.log(bid))) / n

            elif theta_0 is None and theta_1 is None:

                theta_1 = (n * sum(np.log(bid) * np.log(p / (max_cp - p))) - sum(np.log(bid)) * sum(np.log(p / (max_cp - p)))) \
                    / (n * sum(np.log(bid) * np.log(bid)) - sum(np.log(bid))**2)
                theta_0 = (sum(np.log(p / (max_cp - p))) - theta_1 * sum(np.log(bid))) / n

            # Define the probability curve

            return lambda b: max_cp / (1 + np.exp(-theta_0 - theta_1 * np.log(b))) if b != 0 else 0
    
    
class ClickProbabilityLogisticLogShiftModule(ClickProbabilityModule):
    """
    Basic module modeling the bid->click probability relation using a logistic curve composed with
    the natural logarithm and a shift for every segment. Four parameters are used to describe this curve:

        * max_cp -- the upper asymptote.
        * theta_0 -- the "intercept".
        * theta_1 -- the "slope" (thought as the coefficients in the exponent).
        * tau -- the bid "shift" inside the logarithm. Positive value moves the chart
            right.
    The following formula is used:

    .. math::
        \\frac{max\_cp}{1 + e^{-theta_0 - theta_1 \\ln(bid - shift)}}

    Any subset of those parameters can be preset in the initialization. If left as None, max_cp is taken 
    to be twice the historical maximal click probability and theta_0 and theta_1 are found using 
    the least squares method.

    :ivar np.random.RandomState rng: Random number generator.
    :ivar dict prior: Dict with bids, click probabilities and/or some logistic curve params for every segment:

        * list bid -- List of bid values.
        * list p -- List of click probabilities.
        * float theta_0 -- "Intercept" for the logistic curve.
        * float theta_1 -- "Slope" for the logistic curve.
        * float max_cp -- Upper asymptote for the logistic curve.
        * float tau: "Shift" inside the logarithm.
        * str fit_type: Type of fitting to be applied:

            * "lr" -- Linear regression fitting only theta_0 and theta_1,
                while max_cp is estimated from the historical maximum.
            * "gd" -- Gradient descent to find the optimal theta_0, theta_1
                and max_cp.
    :ivar dict[tuple, Callable[[float], float]] segment_func_map: Dict with functions returning
        a probability for every bid for every segment.
    """

    Params = namedtuple('Params', ['bid', 'p', 'theta_0', 'theta_1', 'max_cp', 'tau', 'fit_type'])
    """
    :param list bid: List of bid values
    :param list p: List of click probabilities.
    :param float theta_0: "Intercept" for the logistic curve.
    :param float theta_1: "Slope" for the logistic curve.
    :param float max_cp: Upper asymptote for the logistic curve.
    :param float tau: "Shift" inside the logarithm.
    :param str fit_type: Type of fitting to be applied:

        * "lr" -- Linear regression fitting only theta_0 and theta_1,
            while max_cp is estimated from the historical maximum.
        * "gd" -- Gradient descent to find the optimal theta_0, theta_1
            and max_cp.
    """

    def generate_click_probability_func(self, params):
        """
        :param ClickProbabilityLogisticLogShiftModule.Params params: Params.
        """
        bid = params.bid
        p = params.p
        theta_0 = params.theta_0
        theta_1 = params.theta_1
        max_cp = params.max_cp
        tau = params.tau
        fit_type = params.fit_type

        assert(len(p) == len(bid))

        # p = np.array(p)
        # bid = np.array(bid)
        # p = np.minimum(np.maximum(p, 0.0001), 0.9999)

        # TODO: Code fitting to the real data

        if fit_type == "lr":
            pass

        # Define the probability curve

        return lambda b: max_cp / (1 + np.exp(-theta_0 - theta_1 * np.log(b - tau))) if b - tau > 0 else 0


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":
    
    import unittest


    class TestClickProbabilityConstantModule(unittest.TestCase):
    
        def test_sanity(self):
            print("ClickProbabilityConstantModule class sample run -------------")
            reps = 10

            Params = ClickProbabilityConstantModule.Params
    
            click_prob_model = ClickProbabilityConstantModule(
                prior={
                    (0, 0): Params(bid=1.0, p=0.1),
                    (0, 1): Params(bid=1.0, p=0.3),
                    (1, 0): Params(bid=1.0, p=0.5),
                    (1, 1): Params(bid=1.0, p=0.7)})

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]

            for attr in attributes:
                for r in range(reps):
                    p = click_prob_model.get_cp(np.random.uniform(low=0.0, high=10.0), attr)
                    print("attr={} p={}".format(attr, round(p, 2)))
    
            self.assertTrue(True)
            
            print("")
            
    
    class TestClickProbabilityFunctionModule(unittest.TestCase):
    
        def test_sanity(self):
            print("ClickProbabilityFunctionModule class sample run -------------")
            reps = 10

            Params = ClickProbabilityFunctionModule.Params

            click_prob_model = ClickProbabilityFunctionModule(
                prior={
                    (0, 0): Params(bid=1.0, p=lambda b: min(1.0, max(0.0, b / 10))),
                    (0, 1): Params(bid=1.0, p=lambda b: min(0.5, max(0.0, b / 10))),
                    (1, 0): Params(bid=1.0, p=lambda b: min(1.0, max(0.0, b / 10)) / 2),
                    (1, 1): Params(bid=1.0, p=lambda b: min(0.5, max(0.0, b / 10)) / 2)})

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]

            for attr in attributes:
                for r in range(reps):
                    bid = np.random.uniform(low=0.0, high=10.0)
                    p = click_prob_model.get_cp(bid, attr)
                    print("attr={} bid={} p={}".format(attr, round(bid, 2), round(p, 2)))

            self.assertTrue(True)
            
            print("")
    
    
    class TestClickProbabilityLogisticS1Module(unittest.TestCase):
        
        def test_bid_sensitivity(self):
            print("ClickProbabilityLogisticS1Module class sanity check run (bid sensitivity)-------------")

            Params = ClickProbabilityLogisticS1Module.Params

            click_prob_model = ClickProbabilityLogisticS1Module(
                prior={
                    (0, 0): Params(bid=5.0, p=0.1),
                    (0, 1): Params(bid=5.0, p=0.3),
                    (1, 0): Params(bid=5.0, p=0.5),
                    (1, 1): Params(bid=5.0, p=0.7)})

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            bids = [round(float(i) * 0.2, 2) for i in range(0, 101)]

            for attr in attributes:
                for bid in bids:
                    p = click_prob_model.get_cp(bid, attr)
                    print("attr={} bid={} p={}".format(attr, round(bid, 2), round(p, 2)))
    
            self.assertTrue(True)
            
            print("")
            
    
    class TestClickProbabilityLogisticLogS1Module(unittest.TestCase):
        
        def test_bid_sensitivity(self):
            print("ClickProbabilityLogisticLogS1Module class sanity check run (bid sensitivity)-------------")

            Params = ClickProbabilityLogisticLogS1Module.Params

            click_prob_model = ClickProbabilityLogisticLogS1Module(
                prior={
                    (0, 0): Params(bid=5.0, p=0.1),
                    (0, 1): Params(bid=5.0, p=0.3),
                    (1, 0): Params(bid=5.0, p=0.5),
                    (1, 1): Params(bid=5.0, p=0.7)})

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            bids = [round(float(i) * 0.2, 2) for i in range(0, 101)]

            for attr in attributes:
                for bid in bids:
                    p = click_prob_model.get_cp(bid, attr)
                    print("attr={} bid={} p={}".format(attr, round(bid, 2), round(p, 2)))
    
            self.assertTrue(True)
            
            print("")
            
            
    class TestClickProbabilityLogisticModule(unittest.TestCase):
        
        def test_bid_sensitivity(self):
            print("ClickProbabilityLogisticModule class sanity check run (bid sensitivity)-------------")

            Params = ClickProbabilityLogisticModule.Params

            click_prob_model = ClickProbabilityLogisticModule(
                prior={
                    (0, 0): Params(bid=[4.0, 7.0], p=[0.0, 0.3], theta_0=None, theta_1=None, max_cp=None, fit_type="lr"),
                    (0, 1): Params(bid=[4.0, 7.0], p=[0.3, 0.5], theta_0=None, theta_1=None, max_cp=None, fit_type="lr"),
                    (1, 0): Params(bid=[4.0, 7.0], p=[0.5, 0.7], theta_0=None, theta_1=None, max_cp=None, fit_type="lr"),
                    (1, 1): Params(bid=[4.0, 7.0], p=[0.7, 1.0], theta_0=None, theta_1=None, max_cp=None, fit_type="lr")})

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            bids = [round(float(i) * 0.2, 2) for i in range(0, 101)]

            for attr in attributes:
                for bid in bids:
                    p = click_prob_model.get_cp(bid, attr)
                    print("attr={} bid={} p={}".format(attr, round(bid, 2), round(p, 2)))
    
            self.assertTrue(True)
            
            print("")
            

    class TestClickProbabilityLogisticLogModule(unittest.TestCase):
        
        def test_bid_sensitivity(self):
            print("ClickProbabilityLogisticLogModule class sanity check run (bid sensitivity)-------------")

            Params = ClickProbabilityLogisticLogModule.Params

            click_prob_model = ClickProbabilityLogisticLogModule(
                prior={
                    (0, 0): Params(bid=[4.0, 7.0], p=[0.0, 0.3], theta_0=None, theta_1=None, max_cp=None, fit_type="lr"),
                    (0, 1): Params(bid=[4.0, 7.0], p=[0.3, 0.5], theta_0=None, theta_1=None, max_cp=None, fit_type="lr"),
                    (1, 0): Params(bid=[4.0, 7.0], p=[0.5, 0.7], theta_0=None, theta_1=None, max_cp=None, fit_type="lr"),
                    (1, 1): Params(bid=[4.0, 7.0], p=[0.7, 1.0], theta_0=None, theta_1=None, max_cp=None, fit_type="lr")})

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            bids = [round(float(i) * 0.2, 2) for i in range(0, 101)]

            for attr in attributes:
                for bid in bids:
                    p = click_prob_model.get_cp(bid, attr)
                    print("attr={} bid={} p={}".format(attr, round(bid, 2), round(p, 2)))
            
            print("")


    class TestClickProbabilityLogisticLogShiftModule(unittest.TestCase):

        def test_bid_sensitivity(self):
            print("ClickProbabilityLogisticLogShiftModule class sanity check run (bid sensitivity)-------------")

            Params = ClickProbabilityLogisticLogShiftModule.Params

            click_prob_model = ClickProbabilityLogisticLogShiftModule(
                prior={
                    (0, 0): Params(bid=[], p=[], theta_0=1.0, theta_1=0.5, max_cp=0.5, tau=3.0, fit_type="lr"),
                    (0, 1): Params(bid=[], p=[], theta_0=1.0, theta_1=0.5, max_cp=0.5, tau=6.0, fit_type="lr"),
                    (1, 0): Params(bid=[], p=[], theta_0=1.0, theta_1=0.5, max_cp=0.5, tau=9.0, fit_type="lr"),
                    (1, 1): Params(bid=[], p=[], theta_0=1.0, theta_1=0.5, max_cp=0.5, tau=12.0, fit_type="lr")})

            attributes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            bids = [round(float(i) * 0.2, 2) for i in range(0, 101)]

            for attr in attributes:
                for bid in bids:
                    p = click_prob_model.get_cp(bid, attr)
                    print("attr={} bid={} p={}".format(attr, round(bid, 2), round(p, 2)))

            print("")

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestClickProbabilityConstantModule))
    suite.addTest(unittest.makeSuite(TestClickProbabilityFunctionModule))
    suite.addTest(unittest.makeSuite(TestClickProbabilityLogisticS1Module))
    suite.addTest(unittest.makeSuite(TestClickProbabilityLogisticLogS1Module))
    suite.addTest(unittest.makeSuite(TestClickProbabilityLogisticModule))
    suite.addTest(unittest.makeSuite(TestClickProbabilityLogisticLogModule))
    suite.addTest(unittest.makeSuite(TestClickProbabilityLogisticLogShiftModule))
    unittest.TextTestRunner().run(suite)
