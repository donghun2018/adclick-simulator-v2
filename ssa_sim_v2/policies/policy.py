# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Union, List
import numpy as np

from ssa_sim_v2.simulator.state import StateSet
from ssa_sim_v2.simulator.action import ActionSet, Action
from ssa_sim_v2.simulator.attribute import AttrSet

# ------------------------------------------------------------


class Policy(object):
    """
    Base policy.

    :ivar int policy_id: Policy id. Generally, not needed for the policy to work.
        Used for identification purposes.
    :ivar StateSet state_set: State set -- an object responsible for handling states.
    :ivar ActionSet action_set: Action set -- an object responsible for handling actions.
    :ivar AttrSet attr_set: Attribute set -- an object responsible for handling attributes.
    :ivar object rng: Random number generator.
    :ivar int seed: Seed for the random number generator.
    :ivar Policy.UDP udp: User-defined params.
    :ivar Policy.HTP htp: Hard-tunable params.
    :ivar Policy.STP stp: Soft-tunable params.
    :ivar Policy.IP ip: Inner params.
    :ivar PolicyInitializer policy_initializer: Policy initializer. Normally set to default and only used
        in the initialize method, but exposed as a class attribute to allow dependency injection.
    """

    class UDP(object):
        """
        A class for storing user-defined params -- hard-coded overwrite on all
        other parameters.
        """

        def __init__(self):
            pass

        def update(self, values):
            """
            Updates params based on values in the values dictionary.

            :param dict values: Dictionary with new values.
            """
            for key in values.keys():
                setattr(self, key, values[key])

    class HTP(object):
        """
        A class for storing hard-tunable params -- tuned externally for a vast
        range of bidding entities.
        """

        def __init__(self):
            pass

        def update(self, values):
            """
            Updates params based on values in the values dictionary.

            :param dict values: Dictionary with new values.
            """
            for key in values.keys():
                setattr(self, key, values[key])

    class STP(object):
        """
        A class for storing soft-tunable params -- tuned externally
        for a specific bidding entity (possibly based on a larger dataset
        than inner parameters).
        """

        def __init__(self):
            pass

        def update(self, values):
            """
            Updates params based on values in the values dictionary.

            :param dict values: Dictionary with new values.
            """
            for key in values.keys():
                setattr(self, key, values[key])

    class IP(object):
        """
        A class for storing inner params -- all parameters trained within
        the policy (based on data for a given bidding entity).

        :ivar np.ndarray n: Counts of visits for every pair (state, action).
        """

        def __init__(self):
            pass

        def update(self, values):
            """
            Updates params based on values in the values dictionary.

            :param dict values: Dictionary with new values.
            """
            for key in values.keys():
                setattr(self, key, values[key])

    def __init__(self, state_set, action_set, attr_set, seed=12345, save_history=False):
        """
        :param StateSet state_set: State set -- an object responsible for handling states.
        :param ActionSet action_set: Action set -- an object responsible for handling actions.
        :param AttrSet attr_set: Attribute set -- an object responsible for handling attributes.
        :param int seed: Seed for the random number generator.
        :param bool save_history: Indicates if policy history should be saved
            in the history attribute.
        """

        self.policy_id = None

        self.state_set = state_set
        self.action_set = action_set
        self.attr_set = attr_set
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self.policy_initializer = PolicyInitializer()

        # User-defined params

        self.udp = Policy.UDP()

        # Hard-tunable params

        self.htp = Policy.HTP()

        # Soft-tunable params

        self.stp = Policy.STP()

        # Inner params

        self.ip = Policy.IP()

        # History

        self.save_history = save_history
        # History storing data for the current time step.
        # Data saved for the previous time step gets lost by assumption.
        self.history = dict()

    def initialize(self, params):
        """
        Initializes the policy with given parameters.

        :param dict params: Parameters to be set in the policy.
        """

        self.policy_initializer.params = params
        self.policy_initializer.initialize_policy(self)

    def learn(self, state, data):
        """
        A method that allows the policy to learn based on observations provided
        by the simulator.

        :param StateSet.State state: The state in the previous turn.
        :param Union[pd.DataFrame, dict] data: Input data. Simulator version
            uses the following elements:

            * "action" (Action) -- The action performed by the policy
                in the previous turn.
            * "reward" (float) -- Reward after performing action in the state
                in the previous turn.
            * "info" (dict) -- a dict containing all other data provided
                by the simulator.
            Production version uses pd.DataFrame.
        """

        pass

    def act(self, state, data=None):
        """
        Returns an action given state.

        :param StateSet.State state state: The current state.
        :param Union[pd.DataFrame, dict] data: Input data.
        :return: An action chosen by the policy.
        :rtype: Action
        """

        del state, data  # Used to suppress PyCharm warning

        action = self.action_set.validify_action(Action(bid=1.0))

        # An example of how to store data in the history.
        # It will get saved in the csv file from the simulation.
        self.history.update({"bid": action.bid})

        return action

    def get_htp_definition(self):
        """
        Returns a dictionary with all hard-tunable params definition.
        Keys are parameter names. Values are dictionaries with the following
        fields: "type" -- param type, "min" -- minimal value, "max" -- maximal
        value, "values" -- possible values (if this is set, only these values
        are allowed).

        This method is to be used by external parameter optimizers.

        :return: A definition of hard-tunable params.
        :rtype: dict
        """

        return {}

    def get_stp_definition(self):
        """
        Returns a dictionary with all soft-tunable params definition.
        Keys are parameter names. Values are dictionaries with the following
        fields: "type" -- param type, "min" -- minimal value, "max" -- maximal
        value, "values" -- possible values (if this is set, only these values
        are allowed).

        This method is to be used by external parameter optimizers.

        :return: A definition of soft-tunable params.
        :rtype: dict
        """

        return {}

    def get_learning_data_spec(self, state):
        """
        Get dataset specification structure for data required in policy to run learn method.
        :param ssa_sim_v2.simulator.state.State state: The current state.
        :return: Dictionary with all necessary fields.
        :rtype: dict
        """

        del state  # Used to suppress PyCharm warning

        return {}

    def get_act_data_spec(self, state):
        """
        Get dataset specification structure for data required in policy to run act method.
        :param ssa_sim_v2.simulator.state.State state: The current state.
        :return: Dictionary with all necessary fields.
        :rtype: dict
        """

        del state  # Used to suppress PyCharm warning

        return {}


class PolicyInitializer(object):
    """
    A base class for all policy initializers.
    """

    def __init__(self, params=None):
        """
        :param dict params: A one-level dict with parameter values
            or a two-level dict with parameter groups as keys on the first level
            and parameter names and values as dicts on the second level.
        """
        self.params = params

    def initialize_policy(self, policy):
        """
        Initializes a given policy.

        :param Policy policy: Policy to initialize.
        """

        if self.params is not None:
            for key in self.params.keys():
                if type(self.params[key]) == dict and getattr(policy, key) is not None:
                    getattr(policy, key).update(self.params[key])
                else:
                    setattr(policy, key, self.params[key])
