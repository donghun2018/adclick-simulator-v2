from ssa_sim_v2.policies.policy import Policy
from ssa_sim_v2.simulator.action import Action, ActionSet
from ssa_sim_v2.simulator.attribute import AttrSet
from ssa_sim_v2.simulator.state import StateSet


class Policy2019(Policy):
    """
    Base class for 2019 simulator policy (for students)

    :ivar StateSet state_set: State set -- an object responsible for handling states.
    :ivar ActionSet action_set: Action set -- an object responsible for handling actions.
    :ivar AttrSet attr_set: Attribute set -- an object responsible for handling attributes.
    """

    def __init__(self, state_set, action_set, attr_set, seed=12345, save_history=False):
        """
        :param StateSet state_set: State set -- an object responsible for handling states.
        :param ActionSet action_set: Action set -- an object responsible for handling actions.
        :param AttrSet attr_set: Attribute set -- an object responsible for handling attributes.
        :param int seed: Seed for the random number generator.
        :param bool save_history: Indicates if policy history should be saved
            in the history attribute.
        """

        super().__init__(state_set, action_set, attr_set, seed, save_history)

        # Add any additional class variables here, e.g.:
        # self.my_variable = 1.0
        # self.my_variable_2 = [0.3, 0.4, 0.5]

    def initialize(self, params):
        """
        Initializes the policy with given parameters.

        :param dict params: Parameters to be set in the policy.
        """

        super().initialize(params)

        # Here you can use the following default params to initialize your policy
        # self.stp.cvr_default -- the average historical conversion rate,
        # you can expect the observed average conversion rate to be similar,
        # self.stp.rpv_default -- the average historical value per conversion,
        # you can expect the observed average conversion rate to be similar.
        # You can also use these parameters directly in the learn and act methods.
        # You can also delete this method and/or not use it at all.

    def learn(self, state, data):
        """
        A method that allows the policy to learn based on observations provided
        by the simulator.

        :param StateSet.State state: The state in the previous turn.
        :param Dict data: Dictionary with the following fields:

            * action -- Your original action used in the previous turn.
            * effective_action -- Actual action used by the simulator.
                The original action may need to be adjusted (base bid or modifiers
                clipped to bounds) to be valid.
            * reward -- Overall reward obtained in the previous turn.
            * info -- A dictionary with overall data for the policy:

                * auctions -- number of auctions,
                * clicks -- number of clicks,
                * conversions -- number of conversions,
                * click_probability -- click probability (clicks / auctions),
                * cvr -- conversion rate (conversions / clicks),
                * rpc -- revenue per click (revenue / clicks),
                * cpc -- cost per click (cost / clicks),
                * rpv -- revenue per conversion (revenue / conversions),
                * revenue -- revenue from all conversions,
                * cost -- cost for all clicks,
                * profit -- revenue - cost.
            * attr_info: A dict with data per segment, e.g.
                {
                "gender": {"M": info_for_gender_M, "F": info_for_gender_F, ...},
                "age": {"18-24": info_for_age_18-24, "25-34": info_for_age_25-34, ...},
                ...
                },
                where info_for... has the same form as info but contains data
                only for a given segment.
        """

        pass

    def act(self, state, data=None):
        """
        Returns an action given state.

        :return: An action chosen by the policy.
        """

        # This example method returns a random bid in the range of [min_bid, max_bid]

        b_max = self.action_set.max_bid
        b_min = self.action_set.min_bid
        mod_max = self.action_set.max_mod        # maximum valid value of a modifier
        mod_min = self.action_set.min_mod        # minimum valid value of a modifier

        bid = self.rng.uniform(low=b_min, high=b_max)       # note: use self.rng instead of numpy.random

        action_inc = Action(bid)                            # note: underspecified action (modifiers not defined at all)
        # action_inc = Action(bid, {'gender': {'M': 1.1, 'F': 1.2}})  # underspecified modifiers
        action = self.action_set.validify_action(action_inc)     # this function fills in unspecified modifiers

        # Example how you can access provided default values
        if hasattr(self.stp, "cvr_default"):
            print(self.stp.cvr_default)
        if hasattr(self.stp, "rpv_default"):
            print(self.stp.rpv_default)

        # The following way you can make the simulator save your policy
        # variable values into the output csv file.
        # Save is made after both learn and act methods are invoked.
        self.history.update({"bid": action.bid})

        return action


def test_01_setup():
    """
    sample init init attr, state, action space
    :return:
    """
    names = ['gender', 'age']
    vals = {'gender': ['M', 'F', 'U'],
            'age': ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-*']}
    attr_set = AttrSet(names, vals)

    state_set = StateSet(['date', 'how'], ['discrete', 'discrete'],
                         [['2018-01-01', '2018-01-02'], list(range(168))])

    act_set = ActionSet(attr_set, max_bid=9.99, min_bid=0.01, max_mod=9.0, min_mod=0.1)

    return attr_set, state_set, act_set


def test_one_policy_run():

    # init attr, state, action space
    attr_set, state_set, act_set = test_01_setup()

    # get first state
    s = state_set.make_state({'date': '2018-01-01', 'how': 12})

    # initialize policy
    pol = Policy2019(state_set, act_set, attr_set, seed=9292)
    pol.initialize({"stp": {"cvr_default": 0.02, "rpv_default": 300.0}})
    a = pol.act(s)

    print(a)


if __name__ == "__main__":

    test_01_setup()
    test_one_policy_run()
