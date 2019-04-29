# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":

    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import Dict, List, Tuple
from collections import namedtuple
import numpy as np

from ssa_sim_v2.simulator.action import Action
from ssa_sim_v2.simulator.modules.simulator_module import SimulatorModule

# ------------------------------------------------------------


class VickreyAuctionModule(SimulatorModule):
    """
    Module for performing Vickrey auctions.

    :ivar np.random.RandomState rng: Random number generator.
    """

    Params = namedtuple('Params', [])

    def __init__(self, prior={(0,): Params()}, seed=123):
        """
        :param Dict[tuple, VickreyAuction.Params] prior: Dict with keys for every
            combination of attributes. Values are not used. To indicate that
            there are no segments, {()} can be given as the prior.
        :param int seed: Seed for the random number generator.
        """
        super().__init__(prior, seed)

    def get_auction_results(self, actions):
        """
        Returns Vickrey auction results for given actions.

        :param List[Action] actions: List of actions.
        :return: Dict with attribute combinations as keys. Every value is a list
            of ordered bids for a given combination of attributes. For example,
            assume just one segment and two bids 1.0 with modifier 1.5 and 4.0
            with modifier 0.5. Then the effective bids are 1.5 and 2.0. The result
            will be {(0,): [(1, 2.0), (0, 1.5)]}. The first position in every
            tuple is the policy index, the second is the effective bid for the
            given combination of attributes (intersection of segments).
        :rtype: Dict[Tuple[Union[int, str]], List[Tuple[int, float]]]
        """
        auction_results = dict()
        base_bids_list = []
        for action_index in range(len(actions)):
            base_bids_list.append(actions[action_index].bid)

        for segment in self.prior.keys():
            # For each segment, determine the list of effective modifiers
            actions_effective_modifiers = self.determine_effective_modifiers(actions=actions, segment=segment)
            actions_effective_bids = [bid * modifier for bid, modifier in zip(base_bids_list, actions_effective_modifiers)]
            unsorted_auction = []
            # For each action create a tuple (action_index, action_effective_bid, random_number)
            # the last term will be used for breaking ties and then will be removed for clean output clean_sorted
            for action_index in range(len(actions)):
                unsorted_auction.append((action_index, actions_effective_bids[action_index], self.rng.uniform(0.0, 1000.0)))
            sorted_auction = sorted(unsorted_auction, key=lambda x: (x[1], x[2]), reverse=True)
            clean_sorted = []
            for place in sorted_auction:
                clean_sorted.append((place[0], place[1]))
            auction_results[segment] = clean_sorted

        return auction_results

    def determine_effective_modifiers(self, actions, segment):
        """
        This method accepts a list of actions and a segment and returns a list
        of effective modifiers for each action for that segment.

        :param List[Action] actions: List of actions.
        :param Tuple[int] segment: Attributes combination.
        :return: List of effective modifiers for each action for that segment
        :rtype: List[float]
        """
        # Create dummy array, in this array effective modifiers will be stored. The indices of this list correspond to indices in actions list
        out_array = []
        for action in actions:
            # Set current modifier to 1.0 and iterate through every of segment's dimensions.
            # At each iteration for each dimension retrieve from Action.modifiers
            # a modifier that corresponds to cluster in the segment_type(dimension)
            # that we are currently in, e.g. if segment is (1,2,0) then for Action a
            # we would extract a.modifiers[0][1], a.modifiers[1][2], a.modifiers[2][0]
            # and multiply them all together to create effective modifier
            current_modifier = 1.0
            for i in range(len(segment)):
                current_modifier *= action.modifiers[i][segment[i]]
            out_array.append(current_modifier)
        return out_array


# ==============================================================================
# Unit tests
# ==============================================================================

if __name__ == "__main__":

    import unittest


    class TestVickreyAuctionModule(unittest.TestCase):
        def test_sanity(self):
            print("VickreyAuctionModule class sample run -------------")

            from ssa_sim_v2.simulator.action import Action

            print("Example 1")

            actions = [Action(2.0, [[np.random.uniform(0.5, 1.5) for _ in range(2)]
                                    for _ in range(2)]) for _ in range(3)]

            for action in actions:
                print(action)

            print("")

            vickrey_auction = VickreyAuctionModule(prior={
                (0, 0): VickreyAuctionModule.Params(),
                (0, 1): VickreyAuctionModule.Params(),
                (1, 0): VickreyAuctionModule.Params(),
                (1, 1): VickreyAuctionModule.Params()
            })

            # vickrey_auction.determine_effective_modifiers(actions,segment=(0, 1, 2))

            results = vickrey_auction.get_auction_results(actions=actions)

            for key, value in results.items():
                print("{}: {}".format(key, value))

            print("")

            print("Example 2")

            actions = [Action(2.0, [[np.random.uniform(0.5, 1.5) for _ in range(5)]
                                    for _ in range(7)]) for _ in range(10)]

            for action in actions:
                print(action)

            print("")

            vickrey_auction = VickreyAuctionModule(prior={
                (0, 0, 0): VickreyAuctionModule.Params(),
                (0, 0, 1): VickreyAuctionModule.Params(),
                (0, 0, 2): VickreyAuctionModule.Params(),
                (0, 1, 0): VickreyAuctionModule.Params(),
                (0, 1, 1): VickreyAuctionModule.Params(),
                (0, 1, 2): VickreyAuctionModule.Params(),
                (0, 2, 0): VickreyAuctionModule.Params(),
                (0, 2, 1): VickreyAuctionModule.Params(),
                (0, 2, 2): VickreyAuctionModule.Params(),
                (1, 0, 0): VickreyAuctionModule.Params(),
                (1, 0, 1): VickreyAuctionModule.Params(),
                (1, 0, 2): VickreyAuctionModule.Params(),
                (1, 1, 0): VickreyAuctionModule.Params(),
                (1, 1, 1): VickreyAuctionModule.Params(),
                (1, 1, 2): VickreyAuctionModule.Params(),
                (1, 2, 0): VickreyAuctionModule.Params(),
                (1, 2, 1): VickreyAuctionModule.Params(),
                (1, 2, 2): VickreyAuctionModule.Params(),
                (2, 0, 0): VickreyAuctionModule.Params(),
                (2, 0, 1): VickreyAuctionModule.Params(),
                (2, 0, 2): VickreyAuctionModule.Params(),
                (2, 1, 0): VickreyAuctionModule.Params(),
                (2, 1, 1): VickreyAuctionModule.Params(),
                (2, 1, 2): VickreyAuctionModule.Params(),
                (2, 2, 0): VickreyAuctionModule.Params(),
                (2, 2, 1): VickreyAuctionModule.Params(),
                (2, 2, 2): VickreyAuctionModule.Params(),
            })

            # vickrey_auction.determine_effective_modifiers(actions,segment=(0, 1, 2))

            results = vickrey_auction.get_auction_results(actions=actions)

            for key, value in results.items():
                print("{}: {}".format(key, value))

            print("")

            self.assertTrue(True)


    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestVickreyAuctionModule))
    unittest.TextTestRunner().run(suite)
