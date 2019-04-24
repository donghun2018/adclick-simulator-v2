# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Union, List, Dict

from random import choice

from copy import deepcopy

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ssa_sim_v2.simulator.state import StateSet
from ssa_sim_v2.simulator.action import ActionSet, Action
from ssa_sim_v2.simulator.attribute import AttrSet

from ssa_sim_v2.tools.dict_utils import add_dicts, format_dict

# ------------------------------------------------------------


class CompetitiveDateHowSimulator(object):
    """
    Competitive auction simulator using dates and hours of week in a specified range as states (dates as strings in the
    format yyyy-mm-dd, hour of week as an integer in the range 0-167) and a conversion based revenue
    (a revenue is based on the number of conversions sampled from the number of clicks).

    :ivar StateSet state_set: State set.
    :ivar ActionSet action_set: Action set.
    :ivar AttrSet attr_set: Attribute set.
    :ivar dict modules: Dictionary of modules used to model stochastic variables in the simulator.
    :ivar str date_from: Simulation starting date.
    :ivar str date_from: Simulation final date.
    :ivar float income_share: Optimization type: 1.0 - hotel, 0.x - agency.
    :ivar StateSet.State state: Current time-step.
    :ivar int s_ix: State index.
    :ivar dict internals: Internal variable for storing historical
        internal values of the simulator.
    """

    def __init__(self, state_set, action_set, attr_set, modules,
                 date_from="", date_to="", income_share=1.0):
        """
        :param StateSet state_set: State set.
        :param ActionSet action_set: Action set.
        :param AttrSet attr_set: Attribute set.
        :param dict modules: Dictionary of modules used to
            model stochastic variables in the simulator.
        :param str date_from: Simulation starting date.
        :param str date_from: Simulation final date.
        :param float income_share: Optimization type: 1.0 - hotel, 0.x - agency.
        """
        self.state_set = state_set
        self.action_set = action_set
        self.attr_set = attr_set
        self.modules = modules
        self.date_from = date_from
        self.date_to = date_to
        self.income_share = income_share
        self.internals = dict()  # type: dict

        self.state = self.state_set.make_state({"date": date_from, "how": 0})

    def _next_state(self, state):
        """
        Generates the next time step.

        :param StateSet.State state: State.
        """

        date = state.values.date
        how = state.values.how

        date = datetime.strptime("{} {}".format(date, how % 24), "%Y-%m-%d %H")
        date = date + timedelta(hours=1)
        date = date.strftime("%Y-%m-%d")
        how = (how + 1) % 168

        if date <= self.date_to:
            return self.state_set.make_state({"date": date, "how": how})
        else:
            return None

    def step(self, actions):
        """
        Performs one step of a simulation returning rewards and other info
        for every policy.

        :param List[Action] actions: Actions given by policies.
        :return: List of dictionaries (one dictionary for every policy)
            with the following fields:

            * action -- Action provided by a given policy.
            * effective_action -- Actual action used by the simulator.
                The original action may need to be adjusted (base bid or modifiers
                clipped to bounds) to be valid.
            * reward -- Overall reward for the given policy.
            * info -- A dictionary with overall data for the policy:

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
            * attr_info: A dict with data per segment, e.g.
                {
                "gender": {"M": info_for_gender_M, "F": info_for_gender_F, ...},
                "age": {"18-24": info_for_age_18-24, "25-34": info_for_age_25-34, ...},
                ...
                },
                where info_for... has the same form as info but contains data
                only for a given segment.
        :rtype: List[dict]
        """

        n_pols = len(actions)

        # Fix all actions
        effective_actions = [self.action_set.validify_action(action) for action in actions]
        numerical_actions = [self._prepare_numerical_action(action) for action in effective_actions]

        for a in numerical_actions:
            print(a)

        state = self.state.values

        n_a = self.modules["auctions"].sample(date=state.date, how=state.how)
        attr_auctions = self.modules["auction_attributes"].get_auction_attributes(
            n=n_a, date=state.date, how=state.how)
        auction_results = self.modules["vickrey_auction"].get_auction_results(
            actions=numerical_actions, date=state.date, how=state.how)

        print(auction_results)

        # Prepare an empty response structure

        results = [{} for _ in range(n_pols)]

        info = {
            "auctions": 0,
            "clicks": 0,
            "conversions": 0,
            "positions_sum": 0,
            "revenue": 0.0,
            "cost": 0.0}

        attr_names = self.attr_set.attr_names

        attr_info = {}

        for attr_name in attr_names:
            attr_info[attr_name] = {}
            for attr_value in self.attr_set.attr_sets[attr_name]:
                attr_info[attr_name][attr_value] = deepcopy(info)

        for i in range(n_pols):
            results[i]["action"] = actions[i]
            results[i]["effective_action"] = effective_actions[i]
            results[i]["reward"] = 0.0
            results[i]["info"] = deepcopy(info)
            results[i]["attr_info"] = deepcopy(attr_info)

        # Loop over all attribute combinations

        for attr, attr_n_a in attr_auctions.items():

            print("attr={}".format(attr))

            attr_auction_results = auction_results[attr]

            print("attr auction results={}".format(attr_auction_results))

            pos_to_pols_idxs = np.array([attr_auction_results[i][0] for i in range(n_pols)])

            cp = self.modules["competitive_click_probability"].get_cp(
                auction_results=auction_results, date=state.date, how=state.how, attr=attr)

            print("cp={}".format(cp))

            n_c = self.modules["competitive_clicks"].sample(
                n=attr_n_a, cp=cp, date=state.date, how=state.how, attr=attr)

            print("n_c={}".format(n_c))

            real_cvr = np.array([self.modules["conversion_rate"].get_cvr(
                bid=attr_auction_results[i][1], date=state.date, how=state.how, attr=attr)
                for i in range(n_pols)])
            print("real_cvr={}".format(real_cvr))
            n_v = np.array([self.modules["conversions"].sample(
                num_clicks=n_c[i], cvr=real_cvr[i], date=state.date, how=state.how, attr=attr)
                for i in range(n_pols)])
            print("n_v={}".format(n_v))

            revenue = np.array([self.modules["revenue"].get_revenue(
                num_conversions=n_v[i], date=state.date, how=state.how, attr=attr)
                for i in range(n_pols)])
            print("revenue={}".format(revenue))

            cpc = self.modules["competitive_cpc"].get_cpc(
                auction_results=auction_results, date=state.date, how=state.how, attr=attr)
            print("cpc={}".format(cpc))

            cost = cpc * n_c
            print("cost={}".format(cost))

            profit_is = revenue * self.income_share - cost
            print("profit_is={}".format(profit_is))

            reward = profit_is

            for pos in range(n_pols):
                pol_idx = pos_to_pols_idxs[pos]
                results[pol_idx]["reward"] += reward[pos]
                info = {
                    "auctions": attr_n_a,
                    "clicks": n_c[pos],
                    "conversions": n_v[pos],
                    "positions_sum": (pos + 1) * attr_n_a,
                    "revenue": revenue[pos],
                    "cost": cost[pos]}
                results[pol_idx]["info"] = add_dicts(results[pol_idx]["info"], info)
                attr_cat = self.attr_set.tuples_to_attr(attr)
                for attr_name in attr_names:
                    results[pol_idx]["attr_info"][attr_name][getattr(attr_cat, attr_name)] \
                        = add_dicts(results[pol_idx]["attr_info"][attr_name][getattr(attr_cat, attr_name)], info)

        # Calculate derivative fields

        for pol_idx in range(n_pols):
            self._calculate_derivative_fields(results[pol_idx]["info"])

            bid = actions[pol_idx].bid
            dcpc = bid - results[pol_idx]["info"]["cpc"]
            results[pol_idx]["info"].update(
                {
                    "bid": bid,
                    "dcpc": dcpc
                }
            )

            for attr_name in attr_names:
                for attr_value in self.attr_set.attr_sets[attr_name]:
                    self._calculate_derivative_fields(results[pol_idx]["attr_info"][attr_name][attr_value])

        # TODO: Properly format simulator history
        # # Hist keeping internally
        #
        # prior_auctions = self.modules["auctions"].L.loc[(self.modules["auctions"].L.date == state.date) &
        #                                                 (self.modules["auctions"].L.hour_of_week == state.how),
        #                                                 "auctions"].iloc[0]
        #
        # cp_bid = self.modules["clicks"].p.get_cp(a.bid, state.date, state.how)
        #
        # real_rpv = self.modules["revenue"].models["{}.{}".format(state.date, state.how)].last_rpv
        #
        # real_rpc = real_cvr * real_rpv
        # real_rpc_is = real_rpc * self.income_share
        #
        # expected_profit = prior_auctions * cp_bid * (real_cvr * real_rpv - cpc)
        # expected_profit_is = prior_auctions * cp_bid * (self.income_share * real_cvr * real_rpv - cpc)
        #
        # internals_update = {
        #     "real_cvr": real_cvr,
        #     "real_rpc": real_rpc,
        #     "real_rpc_is": real_rpc_is,
        #     "real_rpv": real_rpv,
        #     "real_rpv_is": real_rpv * self.income_share,
        #     "expected_profit": expected_profit,
        #     "expected_profit_is": expected_profit_is
        # }
        #
        # internals_update.update(info)
        #
        # self.internals.update(internals_update)

        self.state = self._next_state(self.state)

        return results

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
            "average_position": 6.0
        }

        return info

    def get_history(self):
        """
        Returns a copy of the history stored in the simulator.

        :return: A copy of the history stored in the simulator.
        :rtype: dict
        """
        return deepcopy(self.internals)

    def _calculate_derivative_fields(self, info, in_place=True):
        """
        Adds derivative (non-additive) fields to the info dict.

        :param Dict[str, Union[int, float]] info: Dictionary with additive fields:

            * auctions,
            * clicks,
            * conversions,
            * positions_sum,
            * revenue,
            * cost.
        :return: Dictionary with derivative fields added.
        :rtype: Dict[str, Union[int, float]]
        """

        n_a = info["auctions"]
        n_c = info["clicks"]
        n_v = info["conversions"]
        positions_sum = info["positions_sum"]
        revenue = info["revenue"]
        cost = info["cost"]

        click_probability = n_c / n_a if n_a != 0 else 0
        cvr = n_v / n_c if n_c != 0 else 0.0
        average_position = positions_sum / n_a if n_a != 0 else 0.0  # 0.0 is returned by Google Reports in such a case
        rpc = revenue / n_c if n_c != 0 else 0.0
        rpc_is = self.income_share * rpc
        cpc = cost / n_c if n_c != 0 else 0.0
        rpv = revenue / n_v if n_v != 0 else 0.0
        rpv_is = self.income_share * rpv
        cpv = cost / n_v if n_v != 0 else 0.0
        revenue_is = self.income_share * revenue
        profit = revenue - cost
        profit_is = revenue_is - cost

        if in_place:
            result = info
        else:
            result = info.copy()

        result.update(
            {
                "click_probability": click_probability,
                "cvr": cvr,
                "average_position": average_position,
                "rpc": rpc,
                "rpc_is": rpc_is,
                "cpc": cpc,
                "rpv": rpv,
                "rpv_is": rpv_is,
                "cpv": cpv,
                "revenue": revenue,
                "revenue_is": revenue_is,
                "cost": cost,
                "profit": profit,
                "profit_is": profit_is,
            }
        )

        return result

    def _prepare_numerical_action(self, action):
        """
        Prepares a numerical action (with numerical modifiers in the form
        of a list of lists of floats) abusing the Action class definition.
        Must be removed in future versions.

        :param Action action: Action.
        :return: Action with a list of lists of float modifiers instead of dicts.
        :rtype: Action
        """

        modifiers = [[action.modifiers[attr_name][attr_value]
                      for attr_value in self.attr_set.attr_sets[attr_name]]
                     for attr_name in self.attr_set.attr_names]
        numerical_action = Action(action.bid, modifiers)

        return numerical_action


if __name__ == "__main__":
    
    import unittest


    class TestCompetitiveDateHowSimulator(unittest.TestCase):

        def setUp(self):
            from ssa_sim_v2.simulator.modules.auctions.auctions_base_module import AuctionsPoissonModule
            from ssa_sim_v2.simulator.modules.auction_attributes.auction_attributes_base_module import AuctionAttributesModule
            from ssa_sim_v2.simulator.modules.vickrey_auction.vickrey_auction_module import VickreyAuctionModule
            from ssa_sim_v2.simulator.modules.competitive_click_probability.competitive_click_probability_base_module import CompetitiveClickProbabilityTwoClassGeometricModule
            from ssa_sim_v2.simulator.modules.competitive_clicks.competitive_clicks_base_module import CompetitiveClicksBinomialModule
            from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_base_module import ConversionRateFlatModule
            from ssa_sim_v2.simulator.modules.conversions.conversions_base_module import ConversionsBinomialModule
            from ssa_sim_v2.simulator.modules.revenue.revenue_base_module import RevenueGammaNoiseModule
            from ssa_sim_v2.simulator.modules.competitive_cpc.competitive_cpc_base_module import CompetitiveCPCVickreyModule

            from ssa_sim_v2.simulator.modules.auctions.auctions_date_how_module import AuctionsDateHoWModule
            from ssa_sim_v2.simulator.modules.auction_attributes.auction_attributes_date_how_module import AuctionAttributesDateHoWModule
            from ssa_sim_v2.simulator.modules.vickrey_auction.vickrey_auction_date_how_module import VickreyAuctionDateHoWModule
            from ssa_sim_v2.simulator.modules.competitive_click_probability.competitive_click_probability_date_how_module import CompetitiveClickProbabilityDateHoWModule
            from ssa_sim_v2.simulator.modules.competitive_clicks.competitive_clicks_date_how_module import CompetitiveClicksDateHoWModule
            from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_date_how_module import ConversionRateDateHoWModule
            from ssa_sim_v2.simulator.modules.conversions.conversions_date_how_module import ConversionsDateHoWModule
            from ssa_sim_v2.simulator.modules.revenue.revenue_date_how_module import RevenueDateHoWModule
            from ssa_sim_v2.simulator.modules.competitive_cpc.competitive_cpc_date_how_module import CompetitiveCpcDateHoWModule

            self.seed = 1111

            self.date_from = "2018-01-01"
            self.date_to = "2018-01-02"

            self.tmp_df = pd.DataFrame(np.array(range(24)), columns=["hour_of_day"])
            self.tmp_df["key"] = 1
            self.dates = pd.DataFrame(pd.date_range(self.date_from, self.date_to), columns=["date"])
            dates_list = self.dates["date"].tolist()
            self.dates["key"] = 1
            self.dates = pd.merge(self.dates, self.tmp_df, on=["key"], how="left")  # columns: ['date', 'hour_of_day']

            self.dates["hour_of_week"] = pd.to_datetime(self.dates["date"]).dt.dayofweek * 24 + self.dates["hour_of_day"]
            self.dates["date"] = self.dates["date"].dt.strftime("%Y-%m-%d")
            self.dates = self.dates[["date", "hour_of_week"]]

            # Initialize state set
            self.state_set = StateSet(["date", "how"], ["discrete", "discrete"],
                                      [dates_list, list(range(168))])

            # Initialize attribute set
            names = ['gender', 'age']
            vals = {'gender': ['M', 'F', 'U'],
                    'age': ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-*']}
            self.attr_set = AttrSet(names, vals)

            attr_combinations = self.attr_set.get_all_attr_tuples()

            # Initialize action set
            self.action_set = ActionSet(self.attr_set, max_bid=9.99, min_bid=0.01, max_mod=9.0, min_mod=0.1)

            def initialize_priors(params, base_class):
                attr_combinations = list(self.attr_set.get_all_attr_tuples())
                priors = self.dates.copy()
                priors.loc[:, "prior"] = pd.Series([dict.fromkeys(attr_combinations, params)]*len(priors))

                base_classes = self.dates.copy()
                base_classes.loc[:, "base_class"] = base_class

                return priors, base_classes

            # Initialize auctions priors
            module_class = AuctionsPoissonModule
            Params = module_class.Params
            params = Params(auctions=100)
            priors = self.dates.copy()
            priors.loc[:, "prior"] = [{(): params}]*len(priors)
            base_classes = self.dates.copy()
            base_classes.loc[:, "base_class"] = module_class
            self.auctions_priors = priors
            self.auctions_base_classes = base_classes

            # Initialize auction_attributes priors
            module_class = AuctionAttributesModule
            Params = module_class.Params
            params = Params(p=1.0)  # Probabilities are normalized
            self.auction_attributes_priors, self.auction_attributes_base_classes \
                = initialize_priors(params, module_class)

            # Initialize vickrey_auction priors
            module_class = VickreyAuctionModule
            Params = module_class.Params
            params = Params()
            self.vickrey_auction_priors, self.vickrey_auction_base_classes \
                = initialize_priors(params, module_class)

            # Initialize competitive_click_probability priors
            module_class = CompetitiveClickProbabilityTwoClassGeometricModule
            Params = module_class.Params
            params = Params(n_pos=8, p=0.5, q=0.5, r_11=0.6, r_12=0.4, r_2=0.5)
            self.competitive_click_probability_priors, self.competitive_click_probability_base_classes \
                = initialize_priors(params, module_class)

            # Initialize competitive_clicks priors
            module_class = CompetitiveClicksBinomialModule
            Params = module_class.Params
            params = Params(noise_level=0.0, noise_type="multiplicative")
            self.competitive_clicks_priors, self.competitive_clicks_base_classes \
                = initialize_priors(params, module_class)

            # Initialize conversion_rate priors
            module_class = ConversionRateFlatModule
            Params = module_class.Params
            params = Params(cvr=0.02, noise_level=0.0, noise_type="multiplicative")
            self.conversion_rate_priors, self.conversion_rate_base_classes \
                = initialize_priors(params, module_class)

            # Initialize conversions priors
            module_class = ConversionsBinomialModule
            Params = module_class.Params
            params = Params(noise_level=0.0, noise_type="multiplicative")
            self.conversions_priors, self.conversions_base_classes \
                = initialize_priors(params, module_class)

            # Initialize revenue priors
            module_class = RevenueGammaNoiseModule
            Params = module_class.Params
            params = Params(avg_rpv=300.0, noise_level=100.0)
            self.revenue_priors, self.revenue_base_classes = initialize_priors(
                params, module_class)

            # Initialize competitive_cpc priors
            module_class = CompetitiveCPCVickreyModule
            Params = module_class.Params
            params = Params(n_pos=8, fee=0.01)
            self.competitive_cpc_priors, self.competitive_cpc_base_classes = \
                initialize_priors(params, module_class)

            # Module setup for the simulator
            self.mods = \
                {"auctions": AuctionsDateHoWModule(self.auctions_priors,
                                                   self.auctions_base_classes,
                                                   self.seed),
                 "auction_attributes": AuctionAttributesDateHoWModule(self.auction_attributes_priors,
                                                                      self.auction_attributes_base_classes,
                                                                      self.seed),
                 "vickrey_auction": VickreyAuctionDateHoWModule(self.vickrey_auction_priors,
                                                                self.vickrey_auction_base_classes,
                                                                self.seed),
                 "competitive_click_probability": CompetitiveClickProbabilityDateHoWModule(self.competitive_click_probability_priors,
                                                                                           self.competitive_click_probability_base_classes,
                                                                                           self.seed),
                 "competitive_clicks": CompetitiveClicksDateHoWModule(self.competitive_clicks_priors,
                                                                      self.competitive_clicks_base_classes,
                                                                      self.seed),
                 "conversion_rate": ConversionRateDateHoWModule(self.conversion_rate_priors,
                                                                self.conversion_rate_base_classes,
                                                                self.seed),
                 "conversions": ConversionsDateHoWModule(self.conversions_priors,
                                                         self.conversions_base_classes,
                                                         self.seed),
                 "revenue": RevenueDateHoWModule(self.revenue_priors,
                                                 self.revenue_base_classes,
                                                 self.seed),
                 "competitive_cpc": CompetitiveCpcDateHoWModule(self.competitive_cpc_priors,
                                                                self.competitive_cpc_base_classes,
                                                                self.seed)
                 }

            self.simulator = CompetitiveDateHowSimulator(self.state_set, self.action_set, self.attr_set,
                                                         self.mods, self.date_from, self.date_to, income_share=1.0)

        def test_step_method(self):
            print("----------------------------------------")
            print("CompetitiveDateHowSimulator sample step method run")

            N = 48

            for n in range(N):
                print("Round={}".format(n))
                actions = [Action(2.0, {attr_name: {attr_value: np.random.uniform(0.5, 1.5)
                                                    for attr_value in self.attr_set.attr_sets[attr_name]}
                                        for attr_name in self.attr_set.attr_names}) for _ in range(3)]

                for action in actions:
                    print(action)
                print("state={}".format(self.simulator.state))

                results = self.simulator.step(actions)

                print("Results")
                for result in results:
                    print(format_dict(result))
                print("-------")

            self.assertTrue(True)

            print("")

        def test_run_simulation_method(self):
            print("----------------------------------------")
            print("CompetitiveDateHowSimulator sample run_simulation method run")

            N = 48

            for n in range(N):
                print("Round={}".format(n))
                actions = [Action(2.0, {attr_name: {attr_value: np.random.uniform(0.5, 1.5)
                                                    for attr_value in self.attr_set.attr_sets[attr_name]}
                                        for attr_name in self.attr_set.attr_names}) for _ in range(3)]

                for action in actions:
                    print(action)
                print("state={}".format(self.simulator.state))

                results = self.simulator.step(actions)

                # print(results)

            self.assertTrue(True)

            print("")


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCompetitiveDateHowSimulator))
    unittest.TextTestRunner().run(suite)
