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

from ssa_sim_v2.tools.directory_utils import ensure_dir

# ------------------------------------------------------------


class ExperimentResultAggregator:

    def __init__(self, root_directory):
        self.root_directory = root_directory

    def _resample_experiment_df(self, df, policy_name, seed, rule, save=False):
        if df.empty:
            return None

        # Resample experiment
        resampled_df = df.resample(rule, on="datetime")

        df = resampled_df.sum()
        df = df.reset_index()

        counter = resampled_df.count()
        counter = counter.loc[:, ["profit"]]
        counter.columns = [policy_name]

        result = {
            "action_experiment_df": df,
            "counter": counter,
            "policy_name": policy_name
        }

        if save:
            file_name = "resampled_{}_{}_m{}_p{}.pickle".format(rule, policy_name.replace(" ", "_"),
                                                                seed["mod"], seed["pol"])
            file_path = os.path.join(self.root_directory, "resampled_{}".format(rule), file_name)
            ensure_dir(file_path)

            with open(file_path, "wb") as file:
                pickle.dump(result, file)

        return result

    def resample_experiment_df_from_file(self, file_name, rule, save):

        with open(os.path.join(self.root_directory, file_name), "rb") as file:
            result_dict = pickle.load(file)

        batch = result_dict["batch"]
        seed, policy_name, simulator_name, pre_boot_iter = batch

        # Extract and prepare ACTION experiment results
        action_dict = self._resample_experiment_df(df=result_dict["action_experiment_df"],
                                                   policy_name=policy_name,
                                                   seed=seed,
                                                   rule=rule,
                                                   save=save)

        return {
            "policy_name": policy_name,
            "action_experiment_df": action_dict["action_experiment_df"],
            "action_counter": action_dict["counter"]
        }

    def generate_policy_aggregated_statistics(self, df, policy_name):

        # Calculate aggregated statistics
        agg_df = df.groupby(["datetime"])["profit"].agg([np.mean, np.std])

        # Data preparation for mean of policy profit
        policy_mean_result_df = agg_df[["mean"]]
        policy_mean_result_df = policy_mean_result_df.rename(columns={"mean": policy_name})

        # Data preparation for std of policy profit
        policy_std_result_df = agg_df[["std"]]
        policy_std_result_df = policy_std_result_df.rename(columns={"std": policy_name})

        return {
            "mean_df": policy_mean_result_df,
            "std_df": policy_std_result_df
        }

    def aggregate_resampled_repetition_results_from_file(self, rule, policy_order=None, save=False):

        result_list = list()

        for file_name in glob.glob(os.path.join(self.root_directory, "rep_result_*.pickle")):
            with open(file_name, "rb") as file:
                repetition_result = pickle.load(file)

            result_list.append(repetition_result)

        return self.aggregate_resampled_experiment_result(result_list, rule, policy_order, save)

    def aggregate_resampled_experiment_result(self, result_list, rule, policy_order=None, agg_save=False, rep_save=False):
        action_policy_results_dict = defaultdict(list)
        action_policy_counter_dict = dict()

        for policies_results_dict in result_list:
            for policy_name, results_dict in policies_results_dict.items():

                # Extract and prepare experiment results
                results_dict = self._resample_experiment_df(df=results_dict["action_experiment_df"],
                                                            policy_name=policy_name,
                                                            seed=results_dict["seed"],
                                                            rule=rule,
                                                            save=rep_save)

                action_policy_results_dict[policy_name].append(results_dict["action_experiment_df"])
                action_policy_counter_dict[policy_name] = results_dict["counter"]

        # Aggregated statistics
        action_policy_mean_results_df = pd.DataFrame()
        action_policy_std_results_df = pd.DataFrame()
        action_policy_counter_df = pd.DataFrame()

        for policy_name, experiment_df_list in action_policy_results_dict.items():
            policy_experiment_df = pd.concat(experiment_df_list, axis=0)
            # Drop artificial (numerical) index
            policy_experiment_df = policy_experiment_df.reset_index(drop=True)

            # Calculate action statistics
            results_dict = self.generate_policy_aggregated_statistics(df=policy_experiment_df,
                                                                      policy_name=policy_name)

            action_policy_mean_results_df = pd.concat([action_policy_mean_results_df, results_dict["mean_df"]], axis=1)
            action_policy_std_results_df = pd.concat([action_policy_std_results_df, results_dict["std_df"]], axis=1)

        for policy_name, counter_list in action_policy_counter_dict.items():
            action_policy_counter_df = pd.concat([action_policy_counter_df, counter_list], axis=1)

        # Set policy column order if it is needed (helpful for generating visualization)
        if policy_order is not None:
            action_policy_mean_results_df = action_policy_mean_results_df.loc[:, policy_order]
            action_policy_std_results_df = action_policy_std_results_df.loc[:, policy_order]
            action_policy_counter_df = action_policy_counter_df.loc[:, policy_order]

        result = {
            "action_policy_mean_df": action_policy_mean_results_df,
            "action_policy_std_df": action_policy_std_results_df,
            "action_policy_counter_df": action_policy_counter_df
        }

        if agg_save:
            file_name = "final_results_resampled_{}.pickle".format(rule)

            file_path = os.path.join(self.root_directory, "resampled_{}".format(rule), file_name)
            ensure_dir(file_path)

            with open(file_path, "wb") as file:
                pickle.dump(result, file)

        return result
