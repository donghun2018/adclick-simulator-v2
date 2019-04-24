# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":

    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict
import numpy as np

# ------------------------------------------------------------


class Action(object):
    """
    Action class with modifiers.

    :ivar float bid: Bid.
    :ivar dict of dict modifiers: dict of dict of modifiers. Every sub dict
        contains modifiers for a single dimension (e.g. gender, location, device etc.).
        Modifiers are expressed in a multiplicative form, e.g. +30\% is expressed as 1.3.
        The value 1.0 denotes no modifier.
        Example: {bid=1.0, modifiers={'gender': {'F': 1.2, 'M': 1.1, 'U': 1.0}, 
        'age': {'0-19': 0.7, '30-39': 1.1, '60-69': 0.9, '50-59': 0.8, '70-*': 1.0, '20-29': 1.5, '40-49': 1.2}}}

    """
    def __init__(self, bid, modifiers=None):
        """

        :param bid: float
        :param modifiers: if not given, must be validated/initialized against an ActionSet
        """
        self.bid = bid
        self.modifiers = modifiers  # type: Dict[str, Dict[str, float]]

    def __repr__(self):
        if self.modifiers is not None:
            if isinstance(self.modifiers, dict):
                mod_truncated_dict = {k: {k2: np.round(v, 2) for k2, v in d.items()} for k, d in self.modifiers.items()}
                return "{{bid={}, modifiers={}}}".format(self.bid, mod_truncated_dict)
            else:  # To be removed in the future after clean-up
                return "{{bid={}, modifiers={}}}".format(self.bid, [[np.round(v, 2) for v in l] for l in self.modifiers])
        else:   # when modifier is unspecified
            return "{{bid={}, modifiers={}}}".format(self.bid, "None")


class ActionSet(object):
    """
    Action Set class

    provides validator for action
    """

    MOD_DEF_VALUE = 1.0  # default value for modifiers in validify_action

    def __init__(self, attr_set, max_bid, min_bid, max_mod, min_mod):
        """

        :param attr_set: Attribute set object
        :param max_bid: max possible base bid value
        :param min_bid: min possible base bid value
        :param max_mod: max possible modifier value
        :param min_mod: min possible modifier value
        """
        self.attr_set = attr_set
        self.max_bid = max_bid
        self.min_bid = min_bid
        self.max_mod = max_mod
        self.min_mod = min_mod

    def validify_action(self, a, in_place=False):
        """ initialize action as a valid form to the action set.

        implementation: fills all missing modifiers to ActionSet.MOD_DEF_VALUE to create a "valid" action
                        DOES NOT remove unnecessary modifiers not defined in self.attr_set.attr_names

        :param Action a: Action.
        :param bool in_place: if True, param a is modified in-place. Otherwise, a new Action object is returned
        :return: A valid action object, if in_place=False (default); None, otherwise (argument a is updated in-place)
        """
        assert isinstance(a, Action)  # TODO exception handling

        new_mods = {}
        for k in self.attr_set.attr_names:
            new_mod = {k2: ActionSet.MOD_DEF_VALUE for k2 in self.attr_set.attr_sets[k]}
            if a.modifiers is not None and k in a.modifiers.keys():
                new_mod.update(a.modifiers[k])
            new_mods[k] = new_mod

        if in_place:
            a.modifiers = new_mods
            return None
        else:
            return Action(bid=a.bid, modifiers=new_mods)

    def is_valid(self, a):
        """
        returns true if the given action a is "valid" according to this ActionSet
        Validity check
        - bid modifiers are defined for all attributes defined by self.attr_set
        - bid modifiers result in valid bids for all attributes defined by self.attr_set
        :param a: Action
        :return: True, None if valid
                 False, str if invalid. The second str explains the reason why invalid
        """
        base_bid = a.bid
        mod_lists = a.modifiers
        attr_names = self.attr_set.attr_names
        if not len(mod_lists) == len(attr_names):
            return False, "modifier list's length not matching attribute names" # number of attribute names mismatch

        if not self.min_bid <= base_bid:
            return False, "base bid less than min_bid"
        if not base_bid <= self.max_bid:
            return False, "base bid greater than max_bid"

        for k in attr_names:
            try:
                mods = a.modifiers[k]
            except KeyError:
                return False, "modifier does not have key {} defined".format(k)
            mod_list = []
            seg_names = self.attr_set.attr_sets[k]
            for k2 in seg_names:
                try:
                     mod_list.append(mods[k2])
                except KeyError:
                    return False, "modifier for {} does not have segment {} defined".format(k, k2)

            if not all([self.min_mod <= m for m in mod_list]):
                return False, "mod value less than min_mod " # min_mod violated
            if not all([m <= self.max_bid for m in mod_list]):
                return False, "mod value greater than max_mod" # max_mod violated

        return True, None


if __name__ == "__main__":

    # from simulator.attribute import AttrSet

    import attribute

    # sample attrSet
    names = ['gender', 'age']
    vals = {'gender': ['M', 'F', 'U'],
            'age': ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-*']}
    attr_set = attribute.AttrSet(names, vals)

    act_set = ActionSet(attr_set, max_bid=9.99, min_bid=0.01, max_mod=9.0, min_mod=0.1)

    # valid action
    # a1 = Action(1.0, [ [1.1, 1.2, 1.0], [0.7, 1.5, 1.1, 1.2, 0.8, 0.9, 1.0] ] )
    a1 = Action(1.0, {'gender': {'M': 1.1, 'F': 1.2, 'U': 1.0},
                      'age': {'0-19': 0.7, '20-29': 1.5, '30-39': 1.1, '40-49': 1.2, '50-59': 0.8, '60-69': 0.9, '70-*': 1.0}})
    print(act_set.is_valid(a1))

    # invalid action: modifier not fully defined
    a2 = Action(1.0, {'gender': {'M': 1.1, 'F': 1.2, 'U': 1.0},
                      'age': {'0-19': 0.7, '20-29': 1.5, '30-39': 1.1, '40-49': 1.2, '50-59': 0.8, '60-69': 0.9}})
    print(act_set.is_valid(a2))

    # invalid action: less than min_bid found
    a3 = Action(0.00001, {'gender': {'M': 1.1, 'F': 1.2, 'U': 1.0},
                          'age': {'0-19': 0.7, '20-29': 1.5, '30-39': 1.1, '40-49': 1.2, '50-59': 0.8, '60-69': 0.9, '70-*': 1.0}})
    print(act_set.is_valid(a3))

    # invalid action: greater than max_bid found
    a4 = Action(120, {'gender': {'M': 1.1, 'F': 1.2, 'U': 1.0},
                      'age': {'0-19': 0.7, '20-29': 1.5, '30-39': 1.1, '40-49': 1.2, '50-59': 0.8, '60-69': 0.9, '70-*': 1.0}})
    print(act_set.is_valid(a4))

    # invalid action: greater than max_mod found
    a5 = Action(1.0, {'gender': {'M': 1.1, 'F': 1.2, 'U': 1.0},
                      'age': {'0-19': 0.7, '20-29': 1.5, '30-39': 1.1, '40-49': 1.2, '50-59': 0.8, '60-69': 0.9, '70-*': 10.0}})
    print(act_set.is_valid(a5))

    # invalid action: less than min_mod found
    a6 = Action(1.0, {'gender': {'M': 1.1, 'F': 1.2, 'U': 1.0},
                      'age': {'0-19': 0.7, '20-29': 1.5, '30-39': 1.1, '40-49': 1.2, '50-59': 0.8, '60-69': 0.9, '70-*': 0.01}})
    print(act_set.is_valid(a6))

    # check __str__ form of Action
    print(a1)

    # sanity check for validify_action
    a_inc1 = Action(1.0)  # modifier not defined
    print(a_inc1)
    a_inc2 = act_set.validify_action(a_inc1)  # in_place modification of a_inc1
    print(a_inc1, a_inc2)

    # checking in_place flag of validify_action
    a_inc3 = Action(1.0)
    print(a_inc3)
    act_set.validify_action(a_inc3, in_place=True)  # returns a new action (preserves a_inc2)
    print(a_inc3)

    # checking incomplete action fill-ins for a totally missing attribute name
    a_inc4 = Action(1.0, {'gender': {'M': 1.1, 'F': 1.2, 'U': 1.0}})
    print(a_inc4)
    a_inc4_validify = act_set.validify_action(a_inc4)  # in_place modification of a_inc1
    print(a_inc4_validify)

    # checking incomplete action fill-ins for a partially missing attribute name with a totally missing name
    a_inc5 = Action(1.0, {'gender': {'M': 1.1}})
    print(a_inc5)
    a_inc5_validify = act_set.validify_action(a_inc5)  # in_place modification of a_inc1
    print(a_inc5_validify)
