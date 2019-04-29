# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":

    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from collections import defaultdict
import pandas as pd
import numpy as np
import os
import pickle
import glob

# ------------------------------------------------------------


class ExperimentLoader:

    def __init__(self, root_directory):
        self.root_directory = root_directory

    @staticmethod
    def _load_experiment_rep_result(file_name_path):
        with open(file_name_path, "rb") as file:
            rep_result_dict = pickle.load(file)

        # Final structure of the experiment result
        return {
            "action_experiment_df": rep_result_dict["action_experiment_df"],
            "seed": rep_result_dict["seed"],
            "policy_name": rep_result_dict["policy_name"]
        }

    @staticmethod
    def _check_seed_equality(seed_1, seed_2):
        if len(seed_1) != len(seed_2):
            return False
        return np.all([seed_1[key] == seed_2[key] for key in seed_1.keys()])

    def load_experiment_repetition_results_for_each_policy(self, experiment_name):
        """

        :param str experiment_name:
        :return: List of result dicts for every policy name.
        :rtype: dict
        """
        policy_results = defaultdict(list)

        for file_path in glob.glob(os.path.join(self.root_directory, experiment_name, "rep_result_*.pickle")):
            rep_result = self._load_experiment_rep_result(file_path)
            policy_name = rep_result["policy_name"]
            policy_results[policy_name].append(rep_result)

        return policy_results

    def load_experiment_repetition_results_for_defined_policy(self, experiment_name, policy_name):
        """

        :param str experiment_name:
        :param str policy_name:
        :return: List of result dicts for defined policy name.
                    Each element represent experiment repetition with different seeds.
        :rtype: dict
        """
        policy_results = defaultdict(list)

        for file_path in glob.glob(os.path.join(self.root_directory, experiment_name, "rep_result_*.pickle")):
            if policy_name.replace(" ", "_") in file_path:
                rep_result = self._load_experiment_rep_result(file_path)
                policy_name = rep_result["policy_name"]
                policy_results[policy_name].append(rep_result)

        return policy_results

    def get_action_experiment_df_for_defined_policy(self, experiment_name, policy_name, seed_dict):

        policy_results_dict = self.load_experiment_repetition_results_for_defined_policy(experiment_name, policy_name)

        for single_policy_result_dict in policy_results_dict:
            seed = single_policy_result_dict["seed"]

            if self._check_seed_equality(seed, seed_dict):
                return single_policy_result_dict["action_experiment_df"]

        return pd.DataFrame()

    def get_nth_action_experiment_df_for_defined_policy(self, experiment_name, policy_name, repeat_num):
        """

        :param str experiment_name:
        :param str policy_name:
        :param int repeat_num:
        :return:
        """

        policy_results_dict = self.load_experiment_repetition_results_for_defined_policy(experiment_name, policy_name)
        policy_result_dict = policy_results_dict[policy_name][repeat_num]

        return policy_result_dict["action_experiment_df"]

    def get_first_action_experiment_df_for_defined_policy(self, experiment_name, policy_name):

        policy_results_dict = self.load_experiment_repetition_results_for_defined_policy(experiment_name, policy_name)
        first_policy_result_dict = policy_results_dict[policy_name][0]

        return first_policy_result_dict["action_experiment_df"]

    def get_random_action_experiment_df_for_defined_policy(self, experiment_name, policy_name):

        policy_results_list = self.load_experiment_repetition_results_for_defined_policy(experiment_name, policy_name)

        rand_idx = np.random.randint(0, len(policy_results_list[policy_name]), 1)
        policy_result_dict = policy_results_list[policy_name][rand_idx]

        return policy_result_dict["action_experiment_df"]

    def get_experiment_policy_names(self, experiment_name):
        policy_result_dict = self.load_experiment_repetition_results_for_each_policy(experiment_name)

        return list(policy_result_dict.keys())


if __name__ == "__main__":
    loader = ExperimentLoader(os.path.join("data", "output", "20181220_1023"))
    print(loader.get_first_action_experiment_df_for_defined_policy("TEST_NEW_STRUCTURE_4", "Optimal").columns)
