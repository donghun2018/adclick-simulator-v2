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
from scipy.stats import norm
from ssa_sim_v2.simulator.action import Action, ActionSet
from ssa_sim_v2.simulator.attribute import AttrSet
from ssa_sim_v2.simulator.state import StateSet


# ------------------------------------------------------------


class PolicyKnowledgeGradientSI(Policy):
    """
    State-independent (SI) Knowledge Gradient policy.

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

            # Discount factor

            self.discount_factor = 0.95

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
            self.sigma_init = 1000
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
            self.zeta = None
            self.sigma_tilde = None
            self.kg_nu = None

    def __init__(self, state_set, action_set, attr_set, seed=12345, save_history=False):
        """
        :param list state_set: State set.
        :param list action_set: Action set.
        :param int seed: Seed for the random number generator.
        :param bool save_history: Indicates if policy history should be saved
            in the history attribute.
        """

        Policy.__init__(self, state_set, action_set, attr_set, seed=12345, save_history=False)

        self.udp = self.UDP()
        self.htp = self.HTP()
        self.stp = self.STP()
        self.ip = self.IP()

        self.online = True

    def initialize(self, policy_initializer=None):
        """
        Initializes the policy using a policy initializer (dependency injection
        pattern). The policy initializer may be used to test many policy
        parameters settings -- it is enough that the simulator provides
        appropriate policy initializers which set the params.

        :param PolicyInitializer policy_initializer: Policy initializer.
        """
        Policy.initialize(self, policy_initializer)

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
        self.n_actions = round(self.action_set.max_bid)
        # Initialize beliefs
        #n_actions = len(self.action_set)

        self.ip.mu = np.ones([self.n_actions]) * self.stp.mu_init
        self.ip.sigma = np.ones([self.n_actions]) * self.stp.sigma_init

        sigma_tilde_init = self.stp.sigma_init / np.sqrt(1 + (self.stp.sigma_measure / self.stp.sigma_init) ** 2)
        self.ip.sigma_tilde = np.ones([self.n_actions]) * sigma_tilde_init
        self.ip.zeta = np.zeros([self.n_actions])

        self.ip.kg_nu = self.ip.sigma_tilde * (self.ip.zeta * norm.cdf(self.ip.zeta) + norm.pdf(self.ip.zeta))

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

        if data["action"] is None:
            return

        idx = round(data['effective_action'].bid)

        obs_mu = self._get_observation(data)

        # Mu updating for current action
        self.ip.mu[idx] = ((obs_mu * self.ip.sigma[idx] ** 2) + (self.ip.mu[idx] * self.stp.sigma_measure ** 2)) \
            / (self.stp.sigma_measure ** 2 + self.ip.sigma[idx] ** 2)

        # Sigma updating for current action
        self.ip.sigma[idx] = (self.ip.sigma[idx] * self.stp.sigma_measure) \
            / (np.sqrt(self.ip.sigma[idx] ** 2 + self.stp.sigma_measure ** 2))

        # Sigma tilde updating for current action
        self.ip.sigma_tilde[idx] = self.ip.sigma[idx] / np.sqrt(1 + (self.stp.sigma_measure / self.ip.sigma[idx]) ** 2)

        # Zeta and KG nu updating for all actions
        max_mu_idx = np.argsort(self.ip.mu)[-1]
        another_max_mu_value = np.sort(np.delete(self.ip.mu, max_mu_idx))[-1]
        another_max_mu = np.ones([len(self.ip.mu)]) * self.ip.mu[max_mu_idx]
        another_max_mu[max_mu_idx] = another_max_mu_value

        max_mu_idx = np.argsort(self.ip.mu)[-1]
        another_max_mu_value = np.sort(np.delete(self.ip.mu, max_mu_idx))[-1]
        another_max_mu = np.ones([len(self.ip.mu)]) * self.ip.mu[max_mu_idx]
        another_max_mu[max_mu_idx] = another_max_mu_value

        self.ip.zeta = -1 * np.abs((self.ip.mu - another_max_mu) / self.ip.sigma_tilde)

        self.ip.kg_nu = self.ip.sigma_tilde * (self.ip.zeta * norm.cdf(self.ip.zeta) + norm.pdf(self.ip.zeta))

    def act(self, state, data=None):
        """
        Returns an action given state.

        :param State state: The current state.
        :param Union[pd.DataFrame, dict] data: Input data.
        :return: An action chosen by the policy.
        :rtype: Action
        """

        if self.online:
            action_index = dhl.randargmax(self.ip.mu
                                          + self.udp.discount_factor
                                          / (1 - self.udp.discount_factor)
                                          * self.ip.kg_nu, rng=self.rng)
        else:
            action_index = dhl.randargmax(self.ip.kg_nu, rng=self.rng)

        base_bid_amount = self.index_to_bid_amount(action_index)
        action_inc = Action(base_bid_amount)                            # note: underspecified action (modifiers not defined at all)
        # action_inc = Action(bid, {'gender': {'M': 1.1, 'F': 1.2}})  # underspecified modifiers
        action = self.action_set.validify_action(action_inc)     # this function fills in unspecified modifiers

        self.history.update({"bid": action.bid})
        #self.history.update({"bid": self.action_set[action_index].bid})

        return action

    def _get_observation(self, data):
        return data["info"]["profit"] / data["info"]["auctions"] \
            if data["info"]["auctions"] != 0 else 0.0


class PolicyKnowledgeGradientPPASI(PolicyKnowledgeGradientSI):
    """
    State-independent (SI) Knowledge Gradient policy optimizing
    the profit per auction (PPA).
    """

    def _get_observation(self, data):
        return data["info"]["profit"] / data["info"]["auctions"] \
            if data["info"]["auctions"] != 0 else 0.0


class PolicyKnowledgeGradientPSI(PolicyKnowledgeGradientSI):
    """
    State-independent (SI) Knowledge Gradient policy optimizing
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
    pol = PolicyKnowledgeGradientSI(state_set, act_set, attr_set, seed=9292)
    pol.initialize({"stp": {"cvr_default": 0.02, "rpv_default": 300.0}})
    a = pol.act(s)

    print(a)


if __name__ == "__main__":

    test_01_setup()
    test_one_policy_run()