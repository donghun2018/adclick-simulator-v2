# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":

    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from collections import namedtuple
from functools import partial

# ------------------------------------------------------------


class State(object):
    """
    Base state class.

    :ivar str date: Date in the yyyy-mm-dd format.
    """
    def __init__(self, date):
        self.date = date  # type:str


class DateState(State):
    """
    State class with date. The same as the base class but can be used to
    explicitly indicate that this state contains date.

    :ivar str date: Date in the yyyy-mm-dd format.
    """
    def __init__(self, date):
        super().__init__(date)


class DateHowState(State):
    """
    State class with date and hour of week.

    :ivar str date: Date in the yyyy-mm-dd format.
    :ivar int how: Hour of week -- value in the range 0-167.
    """
    def __init__(self, date, how):
        super().__init__(date)
        self.how = how


"""
StateSet class

Donghun Lee 2019

Design Decisions by DH. v1.
- States are dependent on State Set (hence, defined as inner classes)
    - I thought of going for implicit classes, but this needs more discussion, as we may see "State" outside StateSet
- Custom validators are used
    - currently, 'continuous' and 'discrete' types are supported
    - can add more types (like 'date') with custom validators for those
        - did not bother coding them now. someone else can do this by following what I did.
- States can be made using make_state function
    - only after defining a state set
- States can be validated by is_valid function (value boundary checking)
- States can be tested for equality by is_equal function
    - deep copies are considered "equal" as long as all state values are equal
    - DO NOT use is_equal function from a state set with states not from that state set (throws assertion error)

See the script at the end of this file for usage examples

"""


class StateSet(object):
    """
    State Set class, supporting continuous and discrete states
    """
    possible_s_types = ['continuous', 'discrete']

    class State(object):
        """
        internal state class, dependent on stateset.
        """
        def __init__(self, values, types):
            self.values = values
            self.types = types

        def __repr__(self):
            return "(values={} types={})".format(self.values, self.types)

    @staticmethod
    def _valid_continuous(v, min_value_inc, max_value_inc):
        return min_value_inc <= v <= max_value_inc

    @staticmethod
    def _valid_discrete(v, possible_values):
        return v in possible_values

    def __init__(self, s_names, s_types, s_ranges):
        """

        :param s_names: unique name for each state dimension
        :param s_types: one of values in StateSet.possible_s_types.
        :param s_ranges: list of tuples. for each tuple,
                                         if s_types == 'continuous', (min_value, max_value), inclusive
                                         if s_types == 'discrete', a tuple of all possible values
        """
        self.s_names = s_names
        self.StateValue = namedtuple('State_value', s_names)
        self.StateType = namedtuple('State_type', s_names)
        assert len(s_names) == len(s_types) == len(s_ranges)
        self.s_types = dict()
        self.s_ranges = dict()
        self.validators = dict()
        for s_name, s_type, s_range in zip(s_names, s_types, s_ranges):
            assert s_type in StateSet.possible_s_types
            self.s_types[s_name] = s_type
            self.s_ranges[s_name] = s_range
            if s_type == 'continuous':
                self.validators[s_name] = partial(StateSet._valid_continuous,
                                                  min_value_inc=s_range[0],
                                                  max_value_inc=s_range[1])
            elif s_type == 'discrete':
                self.validators[s_name] = partial(StateSet._valid_discrete,
                                                  possible_values=s_range)
            else:
                raise NotImplementedError # state types are defined in possible_s_types, and need own validator fn

    def is_valid(self, s):
        if not isinstance(s.values, self.StateValue):
            return False
        valids = [self.validators[k](v) for k, v in zip(s.values._fields, s.values)]
        return all(valids)

    def make_state(self, values_dict):
        """
        generate state instance
        :param values_dict: {state_name: value} format. all state_names must be found in stateset's s_names
        :return: StateSet.State instance
        """
        values = self.StateValue(**values_dict)
        types = self.StateType(**{k: self.s_types[k] for k in values_dict.keys()})
        return StateSet.State(values, types)

    def is_equal(self, s1, s2):
        assert isinstance(s1.values, self.StateValue)
        assert isinstance(s2.values, self.StateValue)
        equals = [v1 == v2 for v1, v2 in zip(s1.values, s2.values)]
        return all(equals)

if __name__ == "__main__":

    # sample state sets
    ss1 = StateSet(['day_of_week'], ['discrete'], [['M', 'T', 'W', 'R', 'F', 'S', 'U']])
    ss2 = StateSet(['budget'], ['continuous'], [[0, 9999]])
    ss3 = StateSet(['day_of_week', 'budget'], ['discrete', 'continuous'],
                   [['M', 'T', 'W', 'R', 'F', 'S', 'U'], [0, 9999]])

    # discrete state test
    s1_a = ss1.make_state({'day_of_week': 'M'})
    s1_b = ss1.make_state({'day_of_week': 'W'})

    print(ss1.is_valid(s1_a))  # True
    print(ss1.is_valid(s1_b))  # True
    print(ss2.is_valid(s1_a))  # False. testing state from stateset1 whether it is valid for stateset2

    # continuous state test
    s2_a = ss2.make_state({'budget': 25})
    s2_b = ss2.make_state({'budget': -3})
    s2_c = ss2.make_state({'budget': 9999.2})

    print(ss2.is_valid(s2_a))  # True
    print(ss2.is_valid(s2_b))  # False, because this age value is out of bounds
    print(ss2.is_valid(s2_c))  # False, because this age value is also out of bounds
    # TODO: response differentiation between value out of bounds vs state set mismatch.

    # disc/continuous joint state set test
    s3_a = ss3.make_state({'day_of_week': 'M', 'budget': 230.2})
    s3_b = ss3.make_state({'day_of_week': 'F', 'budget': 11.9})

    print(ss3.is_valid(s3_a))  # True
    print(ss3.is_valid(s3_b))  # True
    print(ss1.is_valid(s3_b))  # False, because this state is not from stateset1
                               # (even if there is enough data in s3_b to "define" another state in stateset1)
    print(ss2.is_valid(s3_b))  # False, same as above

    # state equality test
    print(ss3.is_equal(s3_a, s3_b))      # False
    from copy import deepcopy
    s3_a_dup = deepcopy(s3_a)
    print(ss3.is_equal(s3_a, s3_a))      # True, because the state values match
    print(ss3.is_equal(s3_a, s3_a_dup))  # True, because the state values match, even if they are different objects

    # equality test on different state sets
    print(ss2.is_equal(s3_a, s3_b))      # Assertion error.
    # TODO (design decision): do we need to avoid raising Error and have some robust behavior?
