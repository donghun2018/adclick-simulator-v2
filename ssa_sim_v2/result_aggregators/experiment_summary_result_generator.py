# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":

    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import os
import matplotlib.pyplot as plt

# ------------------------------------------------------------


class ExperimentSummaryResultGenerator:

    def __init__(self, root_directory):
        self.root_directory = root_directory
        plt.style.use("seaborn-whitegrid")

    def summary(self, result_dict, save=False):

        action_policy_mean_results_df = result_dict['action_policy_mean_df']
        action_policy_counter_df = result_dict['action_policy_counter_df']

        self._generate_summary_statistics(mean_df=action_policy_mean_results_df,
                                          counter_df=action_policy_counter_df,
                                          save=save)

    def _generate_summary_statistics(self, mean_df, counter_df, save=False):

        # Calculate hourly average profit
        avg_hourly_profit = mean_df.sum(axis=0) / counter_df.astype(float).sum(axis=0)

        result_str = ""
        for policy_name, avg_value in avg_hourly_profit.iteritems():
            result_str += "Policy: {}\n".format(policy_name)

            result_str += "Total profit={}\n".format(
                avg_value * counter_df.astype(float).sum(axis=0).loc[policy_name])
            result_str += "Average hourly profit={}\n".format(avg_value)
            result_str += "\n"

        if save:
            with open(os.path.join(self.root_directory, "Experiment_summary_statistics.txt"), "w") as file:
                file.write(result_str)

        print(result_str)
