# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import List, Dict

from random import choice

from collections import namedtuple
from copy import deepcopy

import numpy as np
import pandas as pd

# ------------------------------------------------------------


class Simulator(object):    
    """
    Base class for all simulators. Step method is a dummy one.

    :ivar list state_set: List of possible states.
    :ivar list action_set: List of valid actions.
    :ivar dict modules: Dictionary of modules used to model stochastic variables in the simulator.
    :ivar int s_ix: State index.
    :ivar dict internals: Internal variable for storing historical
        internal values of the simulator.
    """

    def __init__(self, state_set, action_set, modules):
        """
        :param list state_set: List of possible states.
        :param list action_set: List of valid actions.
        :param dict modules: Dictionary of modules used to model stochastic variables in the simulator.
        """
        self.state_set = state_set  # type: list
        self.action_set = action_set  # type: list
        self.modules = modules  # type: dict
        self.s_ix = 0  # type: int
        self.internals = dict()  # type: dict

    def reset(self):
        """
        Resets the simulator setting the state index to 0.
        """
        
        self.s_ix = 0   # default_cvr init state

    def _next_state(self):
        """
        Moves simulator to the next state.
        """
        self.s_ix += 1
        if self.s_ix >= len(self.state_set):
            self.s_ix = 0

    def step(self, a):
        """
        Performs one step of a simulation returning reward for the given action.

        :param Action a: Action to be performed.

        :return: Tuple of reward (profit after applying the given action) and info dict.
        :rtype: tuple
        """

        assert(a in self.action_set)

        N_A = 10
        N_c = 5

        cpc = 1.0
        rpc = 3.0

        r = (rpc - cpc) * N_c

        # History keeping internally
        self.internals.update({
            "N_A": N_A,
            "N_c": N_c,
            "cpc": cpc,
            "rpc": rpc})

        self._next_state()

        info = {}

        return r, info

    def state(self):
        """
        Returns a copy of the current state.

        :return: action_set copy of the current state.
        :rtype: State
        """
        return deepcopy(self.state_set[self.s_ix])

    def get_history(self):
        """
        Returns a copy of the history stored in the simulator.

        :return: A copy of the history stored in the simulator.
        :rtype: dict
        """
        return deepcopy(self.internals)
    

class SimulatorConstRPC(Simulator):
    """
    Basic auction simulator with auctions, clicks, revenue per click and cost per click modules.
    
    :ivar list state_set: List of possible states.
    :ivar list action_set: List of valid actions.
    :ivar dict modules: Dictionary of modules used to model stochastic variables in the simulator.
    :ivar int s_ix: State index.
    :ivar dict internals: Internal variable for storing historical state values.
    """

    def __init__(self, state_set, action_set, modules):
        """
        :param list state_set: List of possible states.
        :param list action_set: List of valid actions.
        :param dict modules: Dictionary of modules used to model stochastic variables in the simulator.
        """
        Simulator.__init__(self, state_set, action_set, modules)

        assert("auctions" in modules.keys())
        assert("clicks" in modules.keys())
        assert("rpc" in modules.keys())
        assert("cpc" in modules.keys())

    def step(self, a):
        """
        Performs one step of a simulation returning reward for the given action.

        :param Action a: Action to be performed.
        :return: Tuple of reward (profit after applying the given action) and info dict.
        :rtype: tuple
        """

        assert(a in self.action_set)

        N_A = self.modules["auctions"].sample()
        N_c = self.modules["clicks"].sample(n=N_A, bid=a.bid)

        cpc = self.modules["cpc"].get_cpc(a.bid)
        rpc = self.modules["rpc"].get_rpc()

        r = (rpc - cpc) * N_c
        
        info = {"revenue": rpc * N_c,
                "cost": cpc * N_c,
                "num_auction": N_A,
                "num_click": N_c}

        # History keeping internally
        self.internals.update({
            "N_A": N_A,
            "N_c": N_c,
            "cpc": cpc,
            "rpc": rpc})

        self._next_state()

        return r, info


class SimulatorConstRPCHoW(Simulator):
    """
    Auction simulator using hours of week (encoded as a value in the range 0-167) as states 
    and a constant revenue per click for every hour of week.
    
    :ivar list state_set: List of possible states.
    :ivar list action_set: List of valid actions.
    :ivar dict modules: Dictionary of modules used to model stochastic variables in the simulator.
    :ivar int s_ix: State index.
    :ivar dict internals: Internal variable for storing historical state values.
    """

    def __init__(self, state_set, action_set, modules):
        """
        :param list state_set: List of possible states.
        :param list action_set: List of valid actions.
        :param dict modules: Dictionary of modules used to model stochastic variables in the simulator.
        """

        Simulator.__init__(self, state_set, action_set, modules)

        assert("auctions" in modules.keys())
        assert("clicks" in modules.keys())
        assert("rpc" in modules.keys())
        assert("cpc" in modules.keys())

    def step(self, a):
        """
        Performs one step of a simulation returning reward for the given action.

        :param Action a: Action to be performed.
        :return: Tuple of reward (profit after applying the given action) and info dict.
        :rtype: tuple
        """

        assert(a in self.action_set)

        # s_ix is assumed to be hour-of-week
        # In general: (self.s[s_ix].t) % 168 would do the job (assuming t=0 is HoW=0)
        how = self.s_ix

        N_A = self.modules["auctions"].sample(how=how)
        N_c = self.modules["clicks"].sample(n=N_A, bid=a.bid, how=how)

        rpc = self.modules["rpc"].get_rpc(how=how)
        cpc = self.modules["cpc"].get_cpc(a.bid, how=how)
        
        r = (rpc - cpc) * N_c
        
        info = {"revenue": rpc * N_c,
                "cost": cpc * N_c,
                "num_auction": N_A,
                "num_click": N_c}

        # Hist keeping internally
        self.internals.update({
            "N_A": N_A,
            "N_c": N_c,
            "revenue": rpc * N_c,
            "rpc": rpc,
            "cost": cpc * N_c,
            "cpc": cpc})

        self._next_state()

        return r, info


class SimulatorConversionBasedRevenue(Simulator):
    """
    Auction simulator using conversion based revenue (a revenue is based on the number of conversions
    sampled from the number of clicks).
    
    :ivar list state_set: List of possible states.
    :ivar list action_set: List of valid actions.
    :ivar dict modules: Dictionary of modules used to model stochastic variables in the simulator.
    :ivar int s_ix: State index.
    :ivar dict internals: Internal variable for storing historical state values.
    """
    
    def __init__(self, state_set, action_set, modules):
        """
        :param list state_set: List of possible states.
        :param list action_set: List of valid actions.
        :param dict modules: Dictionary of modules used to model stochastic variables in the simulator.
        """

        Simulator.__init__(self, state_set, action_set, modules)

        assert("auctions" in modules.keys())
        assert("clicks" in modules.keys())
        assert("conversions" in modules.keys())
        assert("revenue" in modules.keys())
        assert("cpc" in modules.keys())

    def step(self, a):
        """
        Performs one step of a simulation returning reward for the given action.

        :param Action a: Action to be performed.
        :return: Tuple of reward (profit after applying the given action) and info dict.
        :rtype: tuple
        """

        assert(a in self.action_set)

        N_A = self.modules["auctions"].sample()
        N_c = self.modules["clicks"].sample(n=N_A, bid=a.bid)
        N_v = self.modules["conversions"].sample(n=N_c)

        revenue = self.modules["revenue"].get_revenue(N_v)
        cpc = self.modules["cpc"].get_cpc(a.bid)        

        r = revenue - cpc * N_c
        info = {"revenue": revenue,
                "cost": cpc * N_c,
                "num_auction": N_A,
                "num_click": N_c}

        # Hist keeping internally
        self.internals.update({
            "N_A": N_A,
            "N_c": N_c,
            "N_v": N_v,
            "revenue": revenue,
            "rpc": revenue / N_c,
            "cost": cpc * N_c,
            "cpc": cpc})

        self._next_state()

        return r, info


class SimulatorConversionBasedRevenueHoW(Simulator):
    """
    Auction simulator using hours of week (encoded as a value in the range 0-167) as states 
    and a conversion based revenue (a revenue is based on the number of conversions sampled 
    from the number of clicks).
    
    :ivar list state_set: List of possible states.
    :ivar list action_set: List of valid actions.
    :ivar dict modules: Dictionary of modules used to model stochastic variables in the simulator.
    :ivar int s_ix: State index.
    :ivar dict internals: Internal variable for storing historical state values.
    """
    
    def __init__(self, state_set, action_set, modules):
        """
        :param list state_set: List of possible states.
        :param list action_set: List of valid actions.
        :param dict modules: Dictionary of modules used to model stochastic variables in the simulator.
        """

        Simulator.__init__(self, state_set, action_set, modules)

        assert("auctions" in modules.keys())
        assert("clicks" in modules.keys())
        assert("conversions" in modules.keys())
        assert("revenue" in modules.keys())
        assert("cpc" in modules.keys())

    def step(self, a):
        """
        Performs one step of a simulation returning reward for the given action.

        :param Action a: Action to be performed.
        :return: Tuple of reward (profit after applying the given action)
            and info dict.
        :rtype: tuple
        """

        assert(a in self.action_set)
        how = self.s_ix
        
        N_A = self.modules["auctions"].sample(how=how)
        N_c = self.modules["clicks"].sample(n=N_A, bid=a.bid, how=how)
        N_v = self.modules["conversions"].sample(num_clicks=N_c, how=how)

        revenue = self.modules["revenue"].get_revenue(N_v, how=how)
        cpc = self.modules["cpc"].get_cpc(a.bid, how=how)  

        r = revenue - cpc * N_c
        info = {"revenue": revenue,
                "cost": cpc * N_c,
                "num_auction": N_A,
                "num_click": N_c}

        # Hist keeping internally
        self.internals.update({
            "N_A": N_A,
            "N_c": N_c,
            "N_v": N_v,
            "revenue": revenue,
            "rpc": 0 if N_c == 0 else revenue / N_c,
            "cost": cpc * N_c,
            "cpc": cpc})

        self._next_state()

        return r, info


class SimulatorConversionBasedRevenueDate(Simulator):
    """
    Auction simulator using a series of dates in a specified range as states and a conversion based revenue
    (a revenue is based on the number of conversions sampled from the number of clicks).
    
    :ivar list state_set: List of possible states.
    :ivar list action_set: List of valid actions.
    :ivar dict modules: Dictionary of modules used to model stochastic variables in the simulator.
    :ivar int s_ix: State index.
    :ivar dict internals: Internal variable for storing historical state values.
    :ivar float income_share: Optimization type: 1.0 - hotel, 0.x - OTA.
    """
    
    def __init__(self, state_set, action_set, modules, income_share=1.0):
        """
        :param list state_set: List of possible states.
        :param list action_set: List of valid actions.
        :param dict modules: Dictionary of modules used to model stochastic variables in the simulator.
        :param float income_share: Optimization type: 1.0 - hotel, 0.x - OTA.
        """

        Simulator.__init__(self, state_set, action_set, modules)

        self.hist = []
        self.income_share = income_share
        
        assert("auctions" in modules.keys())
        assert("clicks" in modules.keys())
        assert("conversions" in modules.keys())
        assert("revenue" in modules.keys())
        assert("cpc" in modules.keys())
        # assert("avg_price" in modules.keys())

    def step(self, a):
        """
        Performs one step of a simulation returning reward for the given action.

        :param Action a: Action to be performed.
        :return: Tuple of reward (profit after applying the given action)
            and info dict containing:

            * auctions,
            * clicks,
            * conversions,
            * click_probability,
            * cvr,
            * rpc,
            * rpc_is,
            * cpc,
            * cpc_bid,
            * dcpc,
            * rpv,
            * rpv_is,
            * revenue,
            * revenue_is,
            * cost,
            * profit,
            * profit_is.
        :rtype: tuple
        """

        assert(a in self.action_set)
        state = self.state_set[self.s_ix]

        n_a = self.modules["auctions"].sample(date=state.date)
        n_c = self.modules["clicks"].sample(n=n_a, bid=a.bid, date=state.date)
        real_cvr = self.modules["conversion_rate"].get_cvr(bid=a.bid, date=state.date)
        n_v = self.modules["conversions"].sample(num_clicks=n_c, cvr=real_cvr, date=state.date)

        cpc = self.modules["cpc"].get_cpc(a.bid, date=state.date)

        revenue = self.modules["revenue"].get_revenue(n_v, date=state.date)
        revenue_is = revenue * self.income_share

        rpc = revenue / n_c if n_c != 0 else 0.0
        rpc_is = rpc * self.income_share

        rpv = revenue / n_v if n_v != 0 else 0.0
        rpv_is = rpv * self.income_share

        cost = cpc * n_c

        profit = revenue - cost
        profit_is = revenue * self.income_share - cost

        reward = profit_is

        info = {
            "auctions": n_a,
            "clicks": n_c,
            "conversions": n_v,
            "click_probability": n_c / n_a if n_a != 0 else 0,
            "cvr": n_v / n_c if n_c != 0 else 0.0,
            "rpc": rpc,
            "rpc_is": rpc_is,
            "cpc": cpc,
            "cpc_bid": a.bid,
            "dcpc": a.bid - cpc,
            "rpv": rpv,
            "rpv_is": rpv_is,
            "revenue": revenue,
            "revenue_is": revenue_is,
            "cost": cost,
            "profit": profit,
            "profit_is": profit_is,
        }

        if "avg_price" in self.modules.keys():
            avg_price = self.modules["avg_price"].get_avg_price(date=state.date)
            info["avg_price"] = avg_price

        if "average_position" in self.modules.keys():
            average_position = self.modules["average_position"].get_average_position(
                p=self.modules["click_probability"].get_cp(a.bid, date=state.date),
                date=state.date
            )
            info["average_position"] = average_position

        # Hist keeping internally

        prior_auctions = self.modules["auctions"].L.loc[self.modules["auctions"].L.date == state.date,
                                                        "auctions"].iloc[0]

        cp_bid = self.modules["clicks"].p.get_cp(a.bid, state.date)

        real_rpv = self.modules["revenue"].models[state.date].last_rpv

        real_rpc = real_cvr * real_rpv
        real_rpc_is = real_rpc * self.income_share

        expected_profit = prior_auctions * cp_bid * (real_cvr * real_rpv - cpc)
        expected_profit_is = prior_auctions * cp_bid * (self.income_share * real_cvr * real_rpv - cpc)

        internals_update = {
            "real_cvr": real_cvr,
            "real_rpc": real_rpc,
            "real_rpc_is": real_rpc_is,
            "real_rpv": real_rpv,
            "real_rpv_is": real_rpv * self.income_share,
            "expected_profit": expected_profit,
            "expected_profit_is": expected_profit_is
        }

        internals_update.update(info)

        self.internals.update(internals_update)

        self._next_state()

        return reward, info
    
    def get_empty_info(self):
        """
        Returns a default_cvr lack of activity info for this simulator, as returned
        by the step function. Can be used to make proper initializations
        in policies before the first act.

        :return: Dictionary with default_cvr lack of activity info.
        :rtype: dict
        """

        info = {
            "auctions": 0,
            "clicks": 0,
            "conversions": 0,
            "click_probability": 0.0001,
            "cvr": 0.0,
            "rpc": 0.0,
            "rpc_is": 0.0,
            "cpc": 0.0,
            "cpc_bid": 0.01,
            "dcpc": 0.0,
            "rpv": 0.0,
            "rpv_is": 0.0,
            "revenue": 0.0,
            "revenue_is": 0.0,
            "cost": 0.0,
            "profit": 0.0,
            "profit_is": 0.0,
            "avg_price": 0.0,
            "average_position": 6.0
        }
        
        return info
    

class SimulatorConversionBasedRevenueDateHoW(Simulator):
    """
    Auction simulator using dates and hours of week in a specified range as states (dates as strings in the 
    format yyyy-mm-dd, hour of week as an integer in the range 0-167) and a conversion based revenue 
    (a revenue is based on the number of conversions sampled from the number of clicks).
    
    :ivar list state_set: List of possible states.
    :ivar list action_set: List of valid actions.
    :ivar dict modules: Dictionary of modules used to model stochastic variables in the simulator.
    :ivar int s_ix: State index.
    :ivar dict internals: Internal variable for storing historical state values.
    :ivar float income_share: Optimization type: 1.0 - hotel, 0.x - OTA.
    """
    
    def __init__(self, state_set, action_set, modules, income_share=1.0):
        """
        :param list state_set: List of possible states.
        :param list action_set: List of valid actions.
        :param dict modules: Dictionary of modules used to model stochastic variables in the simulator.
        :param float income_share: Optimization type: 1.0 - hotel, 0.x - OTA.
        """

        Simulator.__init__(self, state_set, action_set, modules)
        self.income_share = income_share

    def step(self, a):
        """
        Performs one step of a simulation returning reward for the given action.

        :param Action a: Action to be performed.
        :return: Tuple of reward (profit after applying the given action)
            and info dict containing:

            * auctions,
            * clicks,
            * conversions,
            * click_probability,
            * cvr,
            * rpc,
            * rpc_is,
            * cpc,
            * cpc_bid,
            * dcpc,
            * rpv,
            * rpv_is,
            * revenue,
            * revenue_is,
            * cost,
            * profit,
            * profit_is.
        :rtype: tuple
        """

        assert a in self.action_set

        state = self.state_set[self.s_ix]
        
        n_a = self.modules["auctions"].sample(date=state.date, how=state.how)
        n_c = self.modules["clicks"].sample(n=n_a, bid=a.bid, date=state.date, how=state.how)
        real_cvr = self.modules["conversion_rate"].get_cvr(bid=a.bid, date=state.date, how=state.how)
        n_v = self.modules["conversions"].sample(num_clicks=n_c, cvr=real_cvr, date=state.date, how=state.how)

        cpc = self.modules["cpc"].get_cpc(a.bid, date=state.date, how=state.how)

        revenue = self.modules["revenue"].get_revenue(n_v, date=state.date, how=state.how)
        revenue_is = revenue * self.income_share

        rpc = revenue / n_c if n_c != 0 else 0.0
        rpc_is = rpc * self.income_share

        rpv = revenue / n_v if n_v != 0 else 0.0
        rpv_is = rpv * self.income_share

        cost = cpc * n_c

        profit = revenue - cost
        profit_is = revenue * self.income_share - cost

        reward = profit_is

        info = {
            "auctions": n_a,
            "clicks": n_c,
            "conversions": n_v,
            "click_probability": n_c / n_a if n_a != 0 else 0,
            "cvr": n_v / n_c if n_c != 0 else 0.0,
            "rpc": rpc,
            "rpc_is": rpc_is,
            "cpc": cpc,
            "cpc_bid": a.bid,
            "dcpc": a.bid - cpc,
            "rpv": rpv,
            "rpv_is": rpv_is,
            "revenue": revenue,
            "revenue_is": revenue_is,
            "cost": cost,
            "profit": profit,
            "profit_is": profit_is,
        }

        if "avg_price" in self.modules.keys():
            avg_price = self.modules["avg_price"].get_avg_price(date=state.date, how=state.how)
            info["avg_price"] = avg_price

        if "average_position" in self.modules.keys():
            average_position = self.modules["average_position"].get_average_position(
                p=self.modules["click_probability"].get_cp(a.bid, date=state.date, how=state.how),
                date=state.date,
                how=state.how
            )
            info["average_position"] = average_position

        # Hist keeping internally

        prior_auctions = self.modules["auctions"].L.loc[(self.modules["auctions"].L.date == state.date) &
                                                        (self.modules["auctions"].L.hour_of_week == state.how),
                                                        "auctions"].iloc[0]

        cp_bid = self.modules["clicks"].p.get_cp(a.bid, state.date, state.how)

        real_rpv = self.modules["revenue"].models["{}.{}".format(state.date, state.how)].last_rpv

        real_rpc = real_cvr * real_rpv
        real_rpc_is = real_rpc * self.income_share

        expected_profit = prior_auctions * cp_bid * (real_cvr * real_rpv - cpc)
        expected_profit_is = prior_auctions * cp_bid * (self.income_share * real_cvr * real_rpv - cpc)

        internals_update = {
            "real_cvr": real_cvr,
            "real_rpc": real_rpc,
            "real_rpc_is": real_rpc_is,
            "real_rpv": real_rpv,
            "real_rpv_is": real_rpv * self.income_share,
            "expected_profit": expected_profit,
            "expected_profit_is": expected_profit_is
        }

        internals_update.update(info)

        self.internals.update(internals_update)

        self._next_state()

        return reward, info

    def get_empty_info(self):
        """
        Returns a default_cvr lack of activity info for this simulator, as returned
        by the step function. Can be used to make proper initializations
        in policies before the first act.

        :return: Dictionary with default_cvr lack of activity info.
        :rtype: dict
        """

        info = {
            "auctions": 0,
            "clicks": 0,
            "conversions": 0,
            "click_probability": 0.0001,
            "cvr": 0.0,
            "rpc": 0.0,
            "rpc_is": 0.0,
            "cpc": 0.0,
            "cpc_bid": 0.01,
            "dcpc": 0.0,
            "rpv": 0.0,
            "rpv_is": 0.0,
            "revenue": 0.0,
            "revenue_is": 0.0,
            "cost": 0.0,
            "profit": 0.0,
            "profit_is": 0.0,
            "avg_price": 0.0,
            "average_position": 6.0
        }

        return info
    

if __name__ == "__main__":
    
    import unittest

    class TestSimulatorConstRPC(unittest.TestCase):
        
        def test_sanity(self):
            print("----------------------------------------")
            print("SimulatorConstRPC sample run")
            
            from ssa_sim_v2.simulator.modules.auctions.auctions_base import AuctionsPoisson
            from ssa_sim_v2.simulator.modules.clicks.clicks_base import ClicksBinomialClickProbFunction
            from ssa_sim_v2.simulator.modules.cpc.cpc_base import CPCFirstPrice
            from ssa_sim_v2.simulator.modules.rpc.rpc_base import RPCUniform
        
            hist_keys = ["s", "a", "r", "env"]
        
            S = namedtuple("State", ["t"])
            A = namedtuple("Action", ["bid"])
        
            N = 10
            Sset = [S(t) for t in range(5)]
            Aset = [A(b) for b in range(10)]
        
            # Load data from csv to find fitted parameter from data
        
            mods = {"auctions": AuctionsPoisson(L=100),
                    "clicks": ClicksBinomialClickProbFunction((lambda b: 0.02)),
                    "cpc": CPCFirstPrice(),
                    "rpc": RPCUniform(low=10, high=100)}
        
            E = SimulatorConstRPC(Sset, Aset, mods)
        
            E.reset()
        
            s = E.state()
        
            hist = []
        
            for n in range(N):
                a = choice(Aset)
        
                r, info = E.step(a)
        
                s2 = E.state()
        
                # Learning
        
                # Hist-keeping
                h = {}
                for k in hist_keys:
                    if k == "s":
                        h[k] = s
                    if k == "a":
                        h[k] = a
                    if k == "r":
                        h[k] = r
                    if k == "env":
                        h[k] = E.get_history()
                hist.append(h)
        
                s = s2
                
            for h in hist:
                print(h)
                print("")
        
            self.assertTrue(True)
            
            print("")
    
    
    class TestSimulatorConstRPCHoW(unittest.TestCase):

        def test_sanity(self):
            print("----------------------------------------")
            print("SimulatorConstRPCHoW sample run")
            
            from ssa_sim_v2.simulator.modules.auctions.auctions_how import AuctionsPoissonHoW
            from ssa_sim_v2.simulator.modules.click_probability.click_probability_how import ClickProbabilityLogisticLogHoW
            from ssa_sim_v2.simulator.modules.clicks.clicks_how import ClicksBinomialClickProbModelHoW
            from ssa_sim_v2.simulator.modules.cpc.cpc_how import CPCBidMinusCpcDiffHoW
            from ssa_sim_v2.simulator.modules.rpc.rpc_how import RPCHistoricalAvgHoW
        
            hist_keys = ["s", "a", "r", "env"]
        
            S = namedtuple("State", ["t"])
            A = namedtuple("Action", ["bid"])
        
            Ssize = 168
            Asize = 5001
            Sset = [S(t) for t in range(Ssize)]
            Aset = [A(round(float(b) / 100, 2)) for b in range(Asize)]
        
            # Initialize auctions prior
            auctions = np.random.exponential(100, size=168)
            
            # Initialize clicks prior
            pc_init = np.random.uniform(low=0.0, high=0.5, size=168)
            bids_init = np.random.uniform(low=0.0, high=20.0, size=168)
            click_prob_model = ClickProbabilityLogisticLogHoW(pc_init, bids_init)
    
            # Initialize rpc prior
            mu_rpc = np.random.uniform(low=10.0, high=50.0, size=168)

            # Initialize cpc prior
            avg_bids = np.random.uniform(high=5.0, size=168)
            avg_cpcs = np.random.uniform(high=avg_bids)
            
            # Module setup for env
            mods = {"auctions": AuctionsPoissonHoW(L=auctions),
                    "clicks": ClicksBinomialClickProbModelHoW(click_prob_model),
                    "rpc": RPCHistoricalAvgHoW(mu_rpc),
                    "cpc": CPCBidMinusCpcDiffHoW(avg_bids, avg_cpcs)}
        
            E = SimulatorConstRPCHoW(Sset, Aset, mods)
        
            E.reset()
        
            s = E.state()
        
            hist = []
        
            N = 168
            for n in range(N):
                a = choice(Aset)
        
                r, info = E.step(a)
        
                s2 = E.state()
        
                # Learning
        
                # Hist-keeping
                h = {}
                for k in hist_keys:
                    if k == "s":
                        h[k] = s
                    if k == "a":
                        h[k] = a
                    if k == "r":
                        h[k] = r
                    if k == "env":
                        h[k] = E.get_history()
                hist.append(h)
        
                s = s2
        
            for h in hist:
                print(h)
                print("")
                
            self.assertTrue(True)
            
            print("")
            
            
    class TestSimulatorConversionBasedRevenueHoW(unittest.TestCase):

        def test_sanity(self):
            print("----------------------------------------")
            print("SimulatorConversionBasedRevenueHoW sample run")
            
            from ssa_sim_v2.simulator.modules.auctions.auctions_how import AuctionsPoissonHoW
            from ssa_sim_v2.simulator.modules.click_probability.click_probability_how import ClickProbabilityLogisticLogHoW
            from ssa_sim_v2.simulator.modules.clicks.clicks_how import ClicksBinomialClickProbModelHoW
            from ssa_sim_v2.simulator.modules.conversions.conversions_how import ConversionsBinomialHoW
            from ssa_sim_v2.simulator.modules.revenue.revenue_how import RevenueConversionBasedHoW
            from ssa_sim_v2.simulator.modules.cpc.cpc_how import CPCBidMinusCpcDiffHoW
        
            hist_keys = ["s", "a", "r", "env"]
        
            S = namedtuple("State", ["t"])
            A = namedtuple("Action", ["bid"])
        
            Ssize = 168
            Asize = 5001
            Sset = [S(t) for t in range(Ssize)]
            Aset = [A(round(float(b) / 100, 2)) for b in range(Asize)]
            
            # Initialize auctions prior
            auctions = np.random.exponential(100, size=168)
            
            # Initialize clicks prior
            pc_init = np.random.uniform(low=0.0, high=0.5, size=168)
            bids_init = np.random.uniform(low=0.0, high=20.0, size=168)
            click_prob_model = ClickProbabilityLogisticLogHoW(pc_init, bids_init)
            
            # Initialize conversions prior
            pv_init = np.random.uniform(low=0.001, high=0.02, size=168)
    
            # Initialize revenue prior
            avg_rpv = np.random.uniform(low=1000.0, high=4000.0, size=168)

            # Initialize cpc prior
            avg_bids = np.random.uniform(high=5.0, size=168)
            avg_cpcs = np.random.uniform(high=avg_bids)
            
            # Module setup for env
            mods = {"auctions": AuctionsPoissonHoW(L=auctions),
                    "clicks": ClicksBinomialClickProbModelHoW(click_prob_model),
                    "conversions": ConversionsBinomialHoW(pv_init),
                    "revenue": RevenueConversionBasedHoW(avg_rpv),
                    "cpc": CPCBidMinusCpcDiffHoW(avg_bids, avg_cpcs)}
        
            E = SimulatorConversionBasedRevenueHoW(Sset, Aset, mods)
        
            E.reset()
        
            s = E.state()
        
            hist = []
        
            N = 168
            for n in range(N):
                a = choice(Aset)
        
                r, info = E.step(a)
        
                s2 = E.state()
        
                # Learning
        
                # Hist-keeping
                h = {}
                for k in hist_keys:
                    if k == "s":
                        h[k] = s
                    if k == "a":
                        h[k] = a
                    if k == "r":
                        h[k] = r
                    if k == "env":
                        h[k] = E.get_history()
                hist.append(h)
        
                s = s2
        
            for h in hist:
                print(h)
                print("")
                
            self.assertTrue(True)
            
            print("")


    class TestSimulatorConversionBasedRevenueDateHoW(unittest.TestCase):

        def setUp(self):
            from ssa_sim_v2.simulator.modules.auctions.auctions_date_how import AuctionsPoissonDateHoW
            from ssa_sim_v2.simulator.modules.click_probability.click_probability_date_how import ClickProbabilityLogisticLogDateHoW
            from ssa_sim_v2.simulator.modules.clicks.clicks_date_how import ClicksBinomialClickProbModelDateHoW
            from ssa_sim_v2.simulator.modules.cpc.cpc_date_how import CPCBidHistoricalAvgCPCDateHoW
            from ssa_sim_v2.simulator.modules.conversions.conversions_date_how import ConversionsBinomialDateHoW
            from ssa_sim_v2.simulator.modules.revenue.revenue_date_how import RevenueConversionBasedDateHoW

            self.S = namedtuple("State", ["date", "how"])
            self.A = namedtuple("Action", ["bid"])

            self.tmp_df = pd.DataFrame(np.array(range(24)), columns=["hour_of_day"])
            self.tmp_df["key"] = 1
            self.dates = pd.DataFrame(pd.date_range('2016-01-01', '2016-01-02'), columns=["date"])
            self.dates["key"] = 1
            self.dates = pd.merge(self.dates, self.tmp_df, on=["key"], how="left")  # columns: ['date', 'hour_of_day']

            self.dates["hour_of_week"] = pd.to_datetime(self.dates["date"]).dt.dayofweek * 24 + self.dates["hour_of_day"]
            self.dates["date"] = self.dates["date"].dt.strftime("%Y-%m-%d")
            self.dates = self.dates[["date", "hour_of_week"]]

            self.Asize = 1001
            self.Ssize = len(self.dates)

            self.Sset = [self.S(date=row[0], how=row[1]) for _, row in self.dates.iterrows()]
            self.Aset = [self.A(bid=(float(b) / 100)) for b in range(self.Asize)]

            self.hist_keys = ["s", "a", "r", "env"]

            # Initialize auctions prior
            self.auctions_df = self.dates.copy()
            self.auctions_df['auctions'] = np.random.exponential(100, size=self.Ssize)

            # Initialize clicks prior
            # Prior DataFrame for click_probability
            self.pc_init_df = self.dates.copy()
            self.pc_init_df['click_probability'] = np.random.uniform(low=0.0, high=0.5, size=self.Ssize)

            # Prior DataFrame for cpc_bid
            self.bids_init_df = self.dates.copy()
            self.bids_init_df['cpc_bid'] = np.random.uniform(low=0.0, high=20.0, size=self.Ssize)

            self.click_prob_model = ClickProbabilityLogisticLogDateHoW(self.pc_init_df,
                                                                       self.bids_init_df)  # Option of clicks modules with PRIOR

            # Initialize cpc prior
            self.cpc_df = self.dates.copy()
            self.avg_bids = np.random.uniform(high=5.0, size=self.Ssize)
            self.avg_cpcs = np.random.uniform(high=self.avg_bids)
            self.cpc_df['average_cpc'] = self.avg_cpcs

            # Initialize revenue prior
            self.revenue_df = self.dates.copy()
            self.revenue_df['value_per_conversion'] = np.random.uniform(low=1000.0, high=4000.0, size=self.Ssize)

            # Initialize conversions prior
            self.conversions_df = self.dates.copy()
            self.conversions_df['conversion_rate'] = np.random.uniform(low=0.001, high=0.02, size=self.Ssize)

            # Module setup for env
            self.mods = {"auctions": AuctionsPoissonDateHoW(L=self.auctions_df),
                         "clicks": ClicksBinomialClickProbModelDateHoW(p=self.click_prob_model),  # p -> probabilistic model
                         "conversions": ConversionsBinomialDateHoW(p=self.conversions_df),  # p -> prior of conversion_rate
                         "revenue": RevenueConversionBasedDateHoW(avg_rpv=self.revenue_df),  # avg_rpv -> prior of value_per_conversion
                         "cpc": CPCBidHistoricalAvgCPCDateHoW(mu_cpc=self.cpc_df)  # mu_cpc -> prior of average_cpc
                         }

            self.E = SimulatorConversionBasedRevenueDateHoW(self.Sset, self.Aset, self.mods)

        def test_sanity(self):
            print("----------------------------------------")
            print("SimulatorConversionBasedRevenueDateHoW sample run")

            hist = []
            N = self.Ssize
            self.E.reset()

            for n in range(N):
                s = self.E.state()
                a = choice(self.Aset)

                r, info = self.E.step(a)

                # Learning
                # Hist-keeping
                h = {}
                for k in self.hist_keys:
                    if k == "s":
                        h[k] = s
                    if k == "a":
                        h[k] = a
                    if k == "r":
                        h[k] = r
                    if k == "env":
                        h[k] = self.E.get_history()
                hist.append(h)

            for h in hist:
                print(h)
                print("")

            self.assertTrue(True)

            print("")

        def test_info_structure_without_avg_price(self):

            N = self.Ssize
            self.E.reset()

            for n in range(N):
                self.E.state()
                a = choice(self.Aset)

                r, info = self.E.step(a)

                self.assertIn("auctions", info.keys())
                self.assertIn("clicks", info.keys())
                self.assertIn("conversions", info.keys())
                self.assertIn("click_probability", info.keys())
                self.assertIn("cpc", info.keys())
                self.assertIn("revenue", info.keys())
                self.assertIn("rpv", info.keys())
                self.assertIn("cost", info.keys())
                self.assertIn("dcpc", info.keys())
                self.assertIn("cpc_bid", info.keys())
                self.assertIn("profit", info.keys())

                self.assertNotIn("avg_price", info.keys())

                self.assertIsNotNone(r)

        def test_info_structure_within_avg_price(self):
            from ssa_sim_v2.simulator.modules.avg_price.avg_price_date_how import AvgPriceHistoricalAvgDateHoW

            avg_price_df = self.dates.copy()
            avg_price_df["avg_price"] = np.random.uniform(low=1000.0, high=4000.0, size=self.Ssize)
            self.mods["avg_price"] = AvgPriceHistoricalAvgDateHoW(avg_price=avg_price_df)

            self.E = SimulatorConversionBasedRevenueDateHoW(self.Sset, self.Aset, self.mods)

            N = self.Ssize
            self.E.reset()

            for n in range(N):
                self.E.state()
                a = choice(self.Aset)

                r, info = self.E.step(a)

                self.assertIn("auctions", info.keys())
                self.assertIn("clicks", info.keys())
                self.assertIn("conversions", info.keys())
                self.assertIn("click_probability", info.keys())
                self.assertIn("cpc", info.keys())
                self.assertIn("revenue", info.keys())
                self.assertIn("rpv", info.keys())
                self.assertIn("cost", info.keys())
                self.assertIn("dcpc", info.keys())
                self.assertIn("cpc_bid", info.keys())
                self.assertIn("profit", info.keys())

                self.assertIn("avg_price", info.keys())

                self.assertIsNotNone(r)


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSimulatorConstRPC))
    suite.addTest(unittest.makeSuite(TestSimulatorConstRPCHoW))
    suite.addTest(unittest.makeSuite(TestSimulatorConversionBasedRevenueHoW))
    suite.addTest(unittest.makeSuite(TestSimulatorConversionBasedRevenueDateHoW))
    unittest.TextTestRunner().run(suite)
