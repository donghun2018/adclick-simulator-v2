# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import numpy as np

from ssa_sim_v2.policies.policy import Policy
from ssa_sim_v2.simulator.action import Action
from ssa_sim_v2.tools import dhl
from ssa_sim_v2.simulator.action import Action, ActionSet
from ssa_sim_v2.simulator.attribute import AttrSet
from ssa_sim_v2.simulator.state import StateSet

# ------------------------------------------------------------


class PolicyThompsonSamplingSI(Policy):
    """
    State-independent (SI) Thompson sampling policy.

    :ivar list state_set: A list of states.
    :ivar list action_set: A list of actions.
    :ivar object rng: Random number generator.
    :ivar int seed: Seed for the random number generator.
    :ivar Policy.UDP udp: User-defined params.
    :ivar Policy.HTP htp: Hard-tunable params.
    :ivar Policy.STP stp: Soft-tunable params.
    :ivar Policy.IP ip: Inner params.
    """

    class UDP(Policy.UDP):
        """
        A class for storing user-defined params -- hard-coded overwrite on all
        other parameters.

        :ivar float min_bid: Minimal allowed bid.
        :ivar float max_bid: Maximal allowed bid.
        """
        def __init__(self):

            Policy.UDP.__init__(self)

            # Min and max bid
            self.min_bid = None
            self.max_bid = None

    class STP(Policy.STP):
        """
        A class for storing soft-tunable params -- tuned externally
        for a specific bidding entity (possibly based on a larger dataset
        than inner parameters).

        :ivar float mu_init: Initial belief for reward value.
        :ivar float sigma_init: Initial uncertainty for the belief.
        :ivar float sigma_measure: Measurement uncertainty.
        """
        def __init__(self):

            Policy.STP.__init__(self)

            self.mu_init = 0.0
            self.sigma_init = 1000.0
            self.sigma_measure = 1.0

    class IP(Policy.IP):
        """
        A class for storing inner params -- all parameters trained within
        the policy (based on data for a given bidding entity).

        :ivar np.ndarray mu: Array of beliefs for the reward for every action.
        :ivar np.ndarray sigma: Array of uncertainties for the reward for every
            action.
        """
        def __init__(self):

            Policy.IP.__init__(self)

            self.mu = None
            self.sigma = None

    def __init__(self, state_set, action_set, attr_set, seed=12345, save_history=False):
        """
        :param StateSet state_set: State set.
        :param ActionSet action_set: Action set.
        :param AttrSet attr_set: Attribute set.
        :param int seed: Seed for the random number generator.
        :param bool save_history: Indicates if policy history should be saved
            in the history attribute.
        """

        Policy.__init__(self, state_set, action_set, attr_set, seed, save_history)

        self.udp = self.UDP()
        self.htp = self.HTP()
        self.stp = self.STP()
        self.ip = self.IP()

    def initialize(self, params):
        """
        Initializes the policy using a policy initializer (dependency injection
        pattern). The policy initializer may be used to test many policy
        parameters settings -- it is enough that the simulator provides
        appropriate policy initializers which set the params.

        :param PolicyInitializer policy_initializer: Policy initializer.
        """
        Policy.initialize(self, params)

        # Apply bounds if defined

        if self.udp.min_bid is not None:
            action_set_temp = []
            for action in self.action_set:
                if action.bid >= self.udp.min_bid:
                    action_set_temp.append(action)

            self.action_set = action_set_temp

        if self.udp.max_bid is not None:
            action_set_temp = []
            for action in self.action_set:
                if action.bid <= self.udp.max_bid:
                    action_set_temp.append(action)

            self.action_set = action_set_temp

        # handy functions to discretize bids
        self.bid_amount_to_index = lambda x: round(x)
        self.index_to_bid_amount = lambda x: float(x)

        # Initialize beliefs
        self.ip.mu = np.array([self.stp.mu_init]*self.bid_amount_to_index(self.action_set.max_bid))
        self.ip.sigma = np.array([self.stp.sigma_init]*self.bid_amount_to_index(self.action_set.max_bid))

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

        Policy.learn(self, state, data)

        if data["action"] is None:
            return

        #idx = self.action_set.index(data["effective_action"])
        idx = round(data['effective_action'].bid)
        obs_mu = self._get_observation(data)

        self.ip.mu[idx] = ((1.0 / self.ip.sigma[idx] ** 2 * self.ip.mu[idx]
                            + 1.0 / self.stp.sigma_measure ** 2 * obs_mu)
                           / (1.0 / self.ip.sigma[idx] ** 2 + 1.0 / self.stp.sigma_measure ** 2))

        self.ip.sigma[idx] = (self.ip.sigma[idx] * self.stp.sigma_measure) \
            / np.sqrt(self.ip.sigma[idx] ** 2 + self.stp.sigma_measure ** 2)

    def act(self, state, data=None):
        """
        Returns an action given state.

        :param State state: The current state.
        :param Union[pd.DataFrame, dict] data: Input data.
        :return: An action chosen by the policy.
        :rtype: Action
        """

        Policy.act(self, state, data)
        try:
            randomized_mu = np.array([self.rng.normal(self.ip.mu[idx], self.ip.sigma[idx])
                                  for idx in range(round(self.action_set.max_bid))])
        except:
            print ('error !!!!!')
            print (self.action_set.max_bid)
            print (len(self.ip.mu))
            print (len(self.ip.sigma))
            print ('length of mu = {}, length of sigma = {}').format(len(self.ip.mu), len(self.ip.sigma))

        action_index = dhl.randargmax(randomized_mu, rng=self.rng)
        base_bid_amount = self.index_to_bid_amount(action_index)
        action_inc = Action(base_bid_amount)           # note: underspecified action (modifiers not defined at all)
        # action_inc = Action(base_bid_amount, {'gender': {'M': 1.1, 'F': 1.2}})  # underspecified modifiers
        action = self.action_set.validify_action(action_inc)     # this function fills in unspecified modifiers

        self.history.update({"bid": action.bid})    # TODO: we should keep not just the base bid, but the entire bid for all attrs

        return action

    def _get_observation(self, data):
        return data["info"]["profit"] / data["info"]["auctions"] \
            if data["info"]["auctions"] != 0 else 0.0


class PolicyThompsonSamplingPPASI(PolicyThompsonSamplingSI):
    """
    State-independent (SI) Thompson sampling policy optimizing
    the profit per auction (PPA).
    """

    def _get_observation(self, data):
        return data["info"]["profit"] / data["info"]["auctions"] \
            if data["info"]["auctions"] != 0 else 0.0


class PolicyThompsonSamplingPSI(PolicyThompsonSamplingSI):
    """
    State-independent (SI) Thompson sampling policy optimizing
    the total profit (P).
    """

    def _get_observation(self, data):
        return data["info"]["profit"]


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
    pol = PolicyThompsonSamplingSI(state_set, act_set, attr_set, seed=9292)
    pol.initialize({"stp": {"cvr_default": 0.02, "rpv_default": 300.0}})
    a = pol.act(s)

    print(a)


if __name__ == "__main__":

    test_01_setup()
    test_one_policy_run()