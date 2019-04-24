"""
Sample bid policy testing script
for ORF418 Spring 2019 course
"""


import numpy as np
import pandas as pd


def simulator_setup_1day():
    """
    This is a tool to set up a simulator and problem definition (state set, action set, and attribute set)
    :return: simulator, state set, action set, attribute set
    """
    from ssa_sim_v2.simulator.modules.auctions.auctions_base_module import AuctionsPoissonModule
    from ssa_sim_v2.simulator.modules.auction_attributes.auction_attributes_base_module import \
        AuctionAttributesModule
    from ssa_sim_v2.simulator.modules.vickrey_auction.vickrey_auction_module import VickreyAuctionModule
    from ssa_sim_v2.simulator.modules.competitive_click_probability.competitive_click_probability_base_module import \
        CompetitiveClickProbabilityTwoClassGeometricModule
    from ssa_sim_v2.simulator.modules.competitive_clicks.competitive_clicks_base_module import \
        CompetitiveClicksBinomialModule
    from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_base_module import ConversionRateFlatModule
    from ssa_sim_v2.simulator.modules.conversions.conversions_base_module import ConversionsBinomialModule
    from ssa_sim_v2.simulator.modules.revenue.revenue_base_module import RevenueGammaNoiseModule
    from ssa_sim_v2.simulator.modules.competitive_cpc.competitive_cpc_base_module import CompetitiveCPCVickreyModule

    from ssa_sim_v2.simulator.modules.auctions.auctions_date_how_module import AuctionsDateHoWModule
    from ssa_sim_v2.simulator.modules.auction_attributes.auction_attributes_date_how_module import \
        AuctionAttributesDateHoWModule
    from ssa_sim_v2.simulator.modules.vickrey_auction.vickrey_auction_date_how_module import \
        VickreyAuctionDateHoWModule
    from ssa_sim_v2.simulator.modules.competitive_click_probability.competitive_click_probability_date_how_module import \
        CompetitiveClickProbabilityDateHoWModule
    from ssa_sim_v2.simulator.modules.competitive_clicks.competitive_clicks_date_how_module import \
        CompetitiveClicksDateHoWModule
    from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_date_how_module import \
        ConversionRateDateHoWModule
    from ssa_sim_v2.simulator.modules.conversions.conversions_date_how_module import ConversionsDateHoWModule
    from ssa_sim_v2.simulator.modules.revenue.revenue_date_how_module import RevenueDateHoWModule
    from ssa_sim_v2.simulator.modules.competitive_cpc.competitive_cpc_date_how_module import \
        CompetitiveCpcDateHoWModule
    from ssa_sim_v2.simulator.competitive_date_how_simulator import CompetitiveDateHowSimulator
    from ssa_sim_v2.simulator.state import StateSet
    from ssa_sim_v2.simulator.action import ActionSet
    from ssa_sim_v2.simulator.attribute import AttrSet

    seed = 1111

    date_from = "2018-01-01"
    date_to = "2018-01-02"

    tmp_df = pd.DataFrame(np.array(range(24)), columns=["hour_of_day"])
    tmp_df["key"] = 1
    dates = pd.DataFrame(pd.date_range(date_from, date_to), columns=["date"])
    dates_list = dates["date"].tolist()
    dates["key"] = 1
    dates = pd.merge(dates, tmp_df, on=["key"], how="left")  # columns: ['date', 'hour_of_day']

    dates["hour_of_week"] = pd.to_datetime(dates["date"]).dt.dayofweek * 24 + dates["hour_of_day"]
    dates["date"] = dates["date"].dt.strftime("%Y-%m-%d")
    dates = dates[["date", "hour_of_week"]]

    # Initialize state set
    state_set = StateSet(["date", "how"], ["discrete", "discrete"],
                              [dates_list, list(range(168))])

    # Initialize attribute set
    names = ['gender', 'age']
    vals = {'gender': ['M', 'F', 'U'],
            'age': ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-*']}
    attr_set = AttrSet(names, vals)

    attr_combinations = attr_set.get_all_attr_tuples()

    # Initialize action set
    action_set = ActionSet(attr_set, max_bid=9.99, min_bid=0.01, max_mod=9.0, min_mod=0.1)

    def initialize_priors(params, base_class):
        attr_combinations = list(attr_set.get_all_attr_tuples())
        priors = dates.copy()
        priors.loc[:, "prior"] = pd.Series([dict.fromkeys(attr_combinations, params)] * len(priors))

        base_classes = dates.copy()
        base_classes.loc[:, "base_class"] = base_class

        return priors, base_classes

    # Initialize auctions priors
    module_class = AuctionsPoissonModule
    Params = module_class.Params
    params = Params(auctions=100)
    priors = dates.copy()
    priors.loc[:, "prior"] = [{(): params}] * len(priors)
    base_classes = dates.copy()
    base_classes.loc[:, "base_class"] = module_class
    auctions_priors = priors
    auctions_base_classes = base_classes

    # Initialize auction_attributes priors
    module_class = AuctionAttributesModule
    Params = module_class.Params
    params = Params(p=1.0)  # Probabilities are normalized
    auction_attributes_priors, auction_attributes_base_classes \
        = initialize_priors(params, module_class)

    # Initialize vickrey_auction priors
    module_class = VickreyAuctionModule
    Params = module_class.Params
    params = Params()
    vickrey_auction_priors, vickrey_auction_base_classes \
        = initialize_priors(params, module_class)

    # Initialize competitive_click_probability priors
    module_class = CompetitiveClickProbabilityTwoClassGeometricModule
    Params = module_class.Params
    params = Params(n_pos=8, p=0.5, q=0.5, r_11=0.6, r_12=0.4, r_2=0.5)
    competitive_click_probability_priors, competitive_click_probability_base_classes \
        = initialize_priors(params, module_class)

    # Initialize competitive_clicks priors
    module_class = CompetitiveClicksBinomialModule
    Params = module_class.Params
    params = Params(noise_level=0.0, noise_type="multiplicative")
    competitive_clicks_priors, competitive_clicks_base_classes \
        = initialize_priors(params, module_class)

    # Initialize conversion_rate priors
    module_class = ConversionRateFlatModule
    Params = module_class.Params
    params = Params(cvr=0.02, noise_level=0.0, noise_type="multiplicative")
    conversion_rate_priors, conversion_rate_base_classes \
        = initialize_priors(params, module_class)

    # Initialize conversions priors
    module_class = ConversionsBinomialModule
    Params = module_class.Params
    params = Params(noise_level=0.0, noise_type="multiplicative")
    conversions_priors, conversions_base_classes \
        = initialize_priors(params, module_class)

    # Initialize revenue priors
    module_class = RevenueGammaNoiseModule
    Params = module_class.Params
    params = Params(avg_rpv=300.0, noise_level=100.0)
    revenue_priors, revenue_base_classes = initialize_priors(
        params, module_class)

    # Initialize competitive_cpc priors
    module_class = CompetitiveCPCVickreyModule
    Params = module_class.Params
    params = Params(n_pos=8, fee=0.01)
    competitive_cpc_priors, competitive_cpc_base_classes = \
        initialize_priors(params, module_class)

    # Module setup for the simulator
    mods = \
        {"auctions": AuctionsDateHoWModule(auctions_priors,
                                           auctions_base_classes,
                                           seed),
         "auction_attributes": AuctionAttributesDateHoWModule(auction_attributes_priors,
                                                              auction_attributes_base_classes,
                                                              seed),
         "vickrey_auction": VickreyAuctionDateHoWModule(vickrey_auction_priors,
                                                        vickrey_auction_base_classes,
                                                        seed),
         "competitive_click_probability": CompetitiveClickProbabilityDateHoWModule(
             competitive_click_probability_priors,
             competitive_click_probability_base_classes,
             seed),
         "competitive_clicks": CompetitiveClicksDateHoWModule(competitive_clicks_priors,
                                                              competitive_clicks_base_classes,
                                                              seed),
         "conversion_rate": ConversionRateDateHoWModule(conversion_rate_priors,
                                                        conversion_rate_base_classes,
                                                        seed),
         "conversions": ConversionsDateHoWModule(conversions_priors,
                                                 conversions_base_classes,
                                                 seed),
         "revenue": RevenueDateHoWModule(revenue_priors,
                                         revenue_base_classes,
                                         seed),
         "competitive_cpc": CompetitiveCpcDateHoWModule(competitive_cpc_priors,
                                                        competitive_cpc_base_classes,
                                                        seed)
         }

    simulator = CompetitiveDateHowSimulator(state_set, action_set, attr_set,
                                                 mods, date_from, date_to, income_share=1.0)

    return simulator, state_set, action_set, attr_set


if __name__ == "__main__":
    """
    This script shows how the bidding policies will interact with the simulator
    The codes are written out for easier understanding and convenient debugging for your policies
    """

    # import policy classes from files
    from policy2019 import Policy2019
    from policy_thompson import PolicyThompsonSamplingSI

    # handy function that initializes all for you
    simulator, state_set, action_set, attr_set = simulator_setup_1day()

    # build "policies" list that contains all bidding policies
    policy1 = Policy2019(state_set, action_set, attr_set, seed=1234)         # this policy is a bare-bone sample policy that bids randomly without learning
    policy2 = PolicyThompsonSamplingSI(state_set, action_set, attr_set, seed=1234)
    policy2.initialize({"stp": {"cvr_default": 0.02, "rpv_default": 300.0}}) # this policy is one of production level policies that needs this extra step
    policies = []
    policies.append(policy1)
    policies.append(policy2)
    policies.append(Policy2019(state_set, action_set, attr_set, seed=9292)) # adding another policy2019 with different seed on-the-fly

    # Simulator will run 24 steps (t=0,1,...,23) (corresponding to 1 simulated day)
    T = 24                          # note that this particular setup limits T up to 48. T>48 will cause an error.
    for t in range(T):
        s = simulator.state
        print("t={} of {}".format(t, T))
        print("  state={}".format(simulator.state))

        actions = []
        for p in policies:
            pol_action = p.act(s)       # each policy responds with a bid
            actions.append(pol_action)
        print("  Actions={}".format(actions))

        results = simulator.step(actions)
        for ix, p in enumerate(policies):
            p.learn(s, results[ix])     # each policy will learn with result
            # note that policy in index ix gets result in index ix. The results can be different



