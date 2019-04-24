"""
Attribute

Donghun Lee 2019

Design decision by DH. v1
- AttrSet (attribute set) class is defined
- Attribute is implicitly defined in the AttrSet class at instantiation
    - this is because an attribute entirely depends on AttrSet instance, especially for its validity check
    - ('age'='10') may be a valid attribute for AttrSet with possible age values ['10', '20', '30'],
      but not one with possible age values ['10-19', '20-29']
    - I left the handle for the custom validators, which can be used to design AttrSet that can see ('age'='10') as
      a valid attribute even with possible age values ['10-19', '20-29']

See the script at the end of this file for a usage example

"""


from collections import namedtuple
from itertools import product


class AttrSet(object):
    """
    Atttribute Set class

    Assumptions
    - each attribute has name. The name is unique across attributes in this set
    - there are finite number of possible attribute values for each attribute
    """

    def __init__(self, attr_names, attr_possible_values, custom_validators=None):
        self.attr_names = attr_names
        self.attr_sets = {k: attr_possible_values[k] for k in attr_names}
        self.Attribute = namedtuple('Attribute', self.attr_names)

    def _make_attr(self, attribute_data_iterable):
        """
        makes an Attribute object from attribute_data_iterable
        :param attribute_data_iterable: dict or list
        :return:
        """
        return self.Attribute(*attribute_data_iterable)

    def get_all_attrs(self):
        """
        returns list of all possible Attribute objects
        :return:
        """
        all_attrs = product(*self.attr_sets.values())
        res = [self._make_attr(a) for a in all_attrs]
        return res

    def get_all_attr_tuples(self):
        """
        requested method to use tuple representations used by many policies.
        warning: tuples cannot be checked using AttrSet.is_valid method
        :return: generator of all attribute tuples
        """
        attr_lists = [self.attr_sets[k] for k in self.attr_names]
        all_attr_tuples = product(*[range(len(l)) for l in attr_lists])
        return all_attr_tuples

    def tuples_to_attr(self, tuple_arg):
        """
        a helper tool to translate tuple into more understandable format
        warning: unsafe. boundary checking is not done.
        :param tuple_arg: a tuple to be translated to Attribute
        :return:
        :rtype: self.Attribute
        """
        attr_data = [self.attr_sets[k][i] for k, i in zip(self.attr_names, tuple_arg)]
        return self._make_attr(attr_data)

    def is_valid(self, a):
        """
        default validator: all attribute values must be found in the attribute set
        :param a: attribute. for example, obtained from get_all_attrs
        :return: True if valid attribute
        """
        assert isinstance(a, self.Attribute)
        res = [getattr(a, k) in self.attr_sets[k] for k in a._fields] # left separate line, for ease of debugging. -DH
        return all(res)


if __name__ == "__main__":
    # this is an example usage -DH

    # sample instantiation
    names = ['gender', 'age']
    vals = {'gender': ['M', 'F', 'U'],
            'age': ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-*']}
    attr_set = AttrSet(names, vals)

    # sanity check.  valid attributes are returned from get_all_attrs function
    attrs = attr_set.get_all_attrs()
    print(attrs)
    print(attr_set.is_valid(attrs[2]))

    # sanity check.  attributes of different value are considered not equal
    print(attrs[1])
    print(attrs[2])
    print(attrs[1] == attrs[2])

    # sanity check.  attributes of same value are considered equal
    from copy import deepcopy
    dup_attr = deepcopy(attrs[1])
    print(attrs[1], id(attrs[1]))
    print(dup_attr, id(dup_attr))
    print(attrs[1] == dup_attr)

    # test get_all_attr_tuples method
    attr_tuples = list(attr_set.get_all_attr_tuples())
    print(attr_tuples)

    # test tuples_to_attr method (all three should be the same)
    print(attr_set.tuples_to_attr((0, 2)))
    print(attr_set.tuples_to_attr((0, 2)).gender)
    print(attr_set.tuples_to_attr(attr_tuples[2]))
    print(attrs[2])

    # Get attr names
    print(attr_set.attr_names)
