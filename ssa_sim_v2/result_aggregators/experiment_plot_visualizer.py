# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":

    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from ssa_sim_v2.tools.directory_utils import ensure_dir
from ssa_sim_v2.result_aggregators.experiment_loader import ExperimentLoader
from ssa_sim_v2.tools.file_data_handler import FileDataHandler

# ------------------------------------------------------------


class SummaryExperimentPlotVisualizer(object):

    def __init__(self, root_directory, chart_directory, experiment_name, state_type):
        self.root_directory = root_directory
        self.series_plot_visualizer = SeriesPlotVisualizer(root_directory=root_directory,
                                                           chart_directory=chart_directory,
                                                           experiment_name=experiment_name,
                                                           state_type=state_type)
        self.bar_plot_visualizer = ProfitBarPlotVisualizer(root_directory=root_directory,
                                                           chart_directory=chart_directory,
                                                           experiment_name=experiment_name,
                                                           state_type=state_type)
        self.rep_result_series_plot_visualizer = RepResultSeriesPlotVisualizer(root_directory=root_directory)
        self.experiment_loader = ExperimentLoader(root_directory=root_directory)

    def generate_summary_profit_series_plots(self, result_dict, save=False):
        self.series_plot_visualizer.visualize(result_dict, save)

    def generate_summary_profit_bar_plots(self, result_dict, save=False):
        self.bar_plot_visualizer.visualize(result_dict, save)


class RepResultExperimentPlotVisualizer(object):

    def __init__(self, root_directory, figsize=(15, 7), save=False):
        self.root_directory = root_directory
        self.figsize = figsize
        self.save = save
        self.rep_result_series_plot_visualizer = RepResultSeriesPlotVisualizer(root_directory, figsize, save)
        self.rep_result_cum_series_plot_visualizer = RepResultCumSeriesPlotVisualizer(root_directory, figsize, save)
        self.experiment_loader = ExperimentLoader(root_directory)
        self.comparison_series_plot_visualizer = PriorSimulatorSeriesPlotVisualizer(root_directory, figsize, save)

        plt.style.use("seaborn-whitegrid")

    def generate_series_bid_rep_result(self, experiment_name):
        self._generate_series_rep_result(series_name="bid", experiment_name=experiment_name)

    def generate_series_profit_rep_result(self, experiment_name):
        """
        Generates series plot of profit obtained from experiment rounds for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of profit (display or save).
        """
        self._generate_series_rep_result(series_name="profit", experiment_name=experiment_name)

    def generate_series_cumulative_bid_rep_result(self, experiment_name, slice_result=None):
        self._generate_cumulative_series_rep_result(series_name="bid", experiment_name=experiment_name, slice_result=slice_result)

    def generate_series_cumulative_profit_rep_result(self, experiment_name, slice_result=None):
        """
        Generates series plot of cumulative profit obtained from experiment rounds for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :param dict slice_result:
        :return: Return generated series plot of profit (display or save).
        """
        self._generate_cumulative_series_rep_result(series_name="profit", experiment_name=experiment_name,
                                                    slice_result=slice_result)

    def generate_series_prior_auctions_rep_result(self, experiment_name):
        """
        Generates series plot of auctions used as a prior in simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of prior auctions (display or save).
        """
        self._generate_series_rep_result(series_name="prior.auctions", experiment_name=experiment_name)

    def generate_series_prior_conversion_rate_rep_result(self, experiment_name, slice_result=None):
        """
        Generates series plot of conversion_rate used as a prior in simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :param dict slice_result:
        :return: Return generated series plot of prior conversion rate (display or save).
        """
        self._generate_series_rep_result(series_name="prior.conversion_rate", experiment_name=experiment_name,
                                         slice_result=slice_result)

    def generate_series_prior_dcpc_rep_result(self, experiment_name):
        """
        Generates series plot of dcpc used as a prior in simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of prior dcpc (display or save).
        """
        self._generate_series_rep_result(series_name="prior.dcpc", experiment_name=experiment_name)

    def generate_series_prior_max_cp_rep_result(self, experiment_name):
        """
        Generates series plot of max_cp used in simulator to calculate number of clicks for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of prior tau (display or save).
        """
        self._generate_series_rep_result(series_name="prior.max_cp", experiment_name=experiment_name)

    def generate_series_prior_tau_rep_result(self, experiment_name):
        """
        Generates series plot of tau used in simulator to calculate number of clicks for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of prior max_cp (display or save).
        """
        self._generate_series_rep_result(series_name="prior.tau", experiment_name=experiment_name)

    def generate_series_prior_theta_1_rep_result(self, experiment_name):
        """
        Generates series plot of theta_1 used in simulator to calculate number of clicks for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of prior theta_1 (display or save).
        """
        self._generate_series_rep_result(series_name="prior.theta_1", experiment_name=experiment_name)

    def generate_series_prior_theta_0_rep_result(self, experiment_name):
        """
        Generates series plot of theta_0 used in simulator to calculate number of clicks for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of prior theta_0 (display or save).
        """
        self._generate_series_rep_result(series_name="prior.theta_0", experiment_name=experiment_name)

    def generate_series_prior_rpv_rep_result(self, experiment_name):
        """
        Generates series plot of rpv used as a prior in simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of prior rpv (display or save).
        """
        self._generate_series_rep_result(series_name="prior.rpv", experiment_name=experiment_name)

    def generate_series_prior_expected_profit_rep_result(self, experiment_name):
        """
        Generates series plot of rpv used as a prior in simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of prior rpv (display or save).
        """
        self._generate_series_rep_result(series_name="prior.expected_profit", experiment_name=experiment_name)

    def generate_series_simulator_auctions_rep_result(self, experiment_name):
        """
        Generates series plot of number of auctions obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of generated auctions (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.auctions", experiment_name=experiment_name)

    def generate_series_simulator_click_probability_rep_result(self, experiment_name):
        """
        Generates series plot for value of click probability obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of click probability (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.click_probability", experiment_name=experiment_name)

    def generate_series_simulator_clicks_rep_result(self, experiment_name):
        """
        Generates series plot of number of clicks obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of generated clicks (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.clicks", experiment_name=experiment_name)

    def generate_series_simulator_conversions_rep_result(self, experiment_name):
        """
        Generates series plot of number of conversions obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of generated conversions (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.conversions", experiment_name=experiment_name)

    def generate_series_simulator_cost_rep_result(self, experiment_name):
        """
        Generates series plot for value of cost obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of generated cost (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.cost", experiment_name=experiment_name)

    def generate_series_simulator_cpc_rep_result(self, experiment_name):
        """
        Generates series plot for value of cpc obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of generated cpc (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.cpc", experiment_name=experiment_name)

    def generate_series_simulator_cpc_bid_rep_result(self, experiment_name):
        """
        Generates series plot for value of cpc bid obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of generated cpc bid (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.cpc_bid", experiment_name=experiment_name)

    def generate_series_simulator_cvr_rep_result(self, experiment_name):
        """
        Generates series plot for value of cvr obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of generated cvr (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.cvr", experiment_name=experiment_name)

    def generate_series_simulator_dcpc_rep_result(self, experiment_name):
        """
        Generates series plot for value of dcpc obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of generated dcpc (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.dcpc", experiment_name=experiment_name)

    def generate_series_simulator_revenue_rep_result(self, experiment_name):
        """
        Generates series plot for value of revenue obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of generated revenue (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.revenue", experiment_name=experiment_name)

    def generate_series_simulator_rpv_rep_result(self, experiment_name):
        """
        Generates series plot for value of rpv obtained from simulator for defined experiment spec name.

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of generated rpv (display or save).
        """
        self._generate_series_rep_result(series_name="simulator.rpv", experiment_name=experiment_name)

    def generate_comparison_series_auctions_rep_result(self, experiment_name):
        """

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of profit (display or save).
        """
        self._generate_comparison_series_rep_result(series_name="auctions", experiment_name=experiment_name)

    def generate_comparison_series_clicks_rep_result(self, experiment_name):
        """

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of profit (display or save).
        """
        self._generate_comparison_series_rep_result(series_name="clicks", experiment_name=experiment_name)

    def generate_comparison_series_conversion_rate_rep_result(self, experiment_name):
        """

        :param str experiment_name: Name of the experiment spec name.
        :return: Return generated series plot of profit (display or save).
        """
        self._generate_comparison_series_rep_result(series_name="conversion_rate", experiment_name=experiment_name)

    def _generate_series_rep_result(self, series_name, experiment_name, slice_result=None):
        """

        :param str series_name:
        :param str experiment_name:
        :return:
        """
        policy_result_dict = self.experiment_loader.load_experiment_repetition_results_for_each_policy(experiment_name)
        self.rep_result_series_plot_visualizer.visualize(policy_result_dict=policy_result_dict,
                                                         series_name=series_name,
                                                         experiment_name=experiment_name,
                                                         slice_result=slice_result)

    def _generate_comparison_series_rep_result(self, series_name, experiment_name, slice_result=None):
        """

        :param str series_name:
        :param str experiment_name:
        :return:
        """
        policy_result_dict = self.experiment_loader.load_experiment_repetition_results_for_each_policy(experiment_name)
        self.comparison_series_plot_visualizer.visualize(policy_result_dict=policy_result_dict,
                                                         series_name=series_name,
                                                         experiment_name=experiment_name,
                                                         slice_result=slice_result)

    def _generate_cumulative_series_rep_result(self, series_name, experiment_name, slice_result=None):
        """

        :param str series_name:
        :param str experiment_name:
        :return:
        """
        policy_result_dict = self.experiment_loader.load_experiment_repetition_results_for_each_policy(experiment_name)
        self.rep_result_cum_series_plot_visualizer.visualize(policy_result_dict=policy_result_dict,
                                                             series_name=series_name,
                                                             experiment_name=experiment_name,
                                                             slice_result=slice_result)


class PlotVisualizer(object):

    def __init__(self, root_directory, chart_directory=None, experiment_name=None):
        self.root_directory = root_directory
        self.chart_directory = chart_directory
        self.experiment_name = experiment_name

    def visualize(self, **kwargs):
        pass


class RepResultSeriesPlotVisualizer(PlotVisualizer):

    def __init__(self, root_directory, figsize=(15, 7), save=False):
        PlotVisualizer.__init__(self, root_directory)
        self.figsize = figsize
        self.save = save

    def visualize(self, policy_result_dict, series_name, experiment_name, slice_result=None):
        """

        :param policy_result_dict:
        :param series_name:
        :param experiment_name:
        :param slice_result:
        :return:
        """
        df_list = list()
        extracted_index = None

        for policy_name, rep_result_dict_list in policy_result_dict.items():
            for rep_result_dict in rep_result_dict_list:
                seed_dict = rep_result_dict["seed"]

                action_df = rep_result_dict["action_experiment_df"]
                if extracted_index is None:
                    extracted_index = action_df["datetime"]
                action_df["policy_name"] = policy_name
                action_df["seed_mod"] = seed_dict["mod"]
                action_df["seed_pol"] = seed_dict["pol"]
                df_list.append(action_df)

        final_df = pd.concat(df_list, axis=0, ignore_index=True)  # type: pd.DataFrame

        if series_name not in final_df.columns:
            raise ValueError("Incorrect value of the series. There isn't series {} in experiment result DataFrame".format(series_name))

        for group_idx, group_df in final_df.groupby(['seed_mod', 'seed_pol']):
            policy_series_list = list()
            policy_name_list = list()

            for policy_name, policy_group in group_df.groupby(["policy_name"])[series_name]:
                policy_name_list.append(policy_name)
                policy_series_list.append(policy_group.reset_index(drop=True))

            result_df = pd.concat(policy_series_list, axis=1)

            result_df = result_df.set_index(extracted_index)  # type: pd.DataFrame

            result_df.columns = policy_name_list

            if slice_result is not None:
                result_df = result_df.loc[slice_result["start"]:slice_result["end"], :]

            ax = result_df.plot(figsize=self.figsize, alpha=0.5)

            ax.set_ylabel(series_name.capitalize().replace("_", " "))
            ax.set_title("Experiment repetition results for {}".format(series_name))
            # plt.tight_layout()
            x0, x1, y0, y1 = plt.axis()
            plot_margin = 2.25
            plt.axis((x0 - plot_margin,
                      x1 + plot_margin,
                      y0 - plot_margin,
                      y1 + plot_margin))

            if self.save:
                file_name = "experiment_repetition_results_[{}].png".format(series_name)
                plt.savefig(os.path.join(self.root_directory, experiment_name, file_name), bbox_inches="tight")
                plt.close()
            else:
                plt.show()


class PriorSimulatorSeriesPlotVisualizer(PlotVisualizer):

    def __init__(self, root_directory, figsize=(15, 7), save=False):
        PlotVisualizer.__init__(self, root_directory)
        self.figsize = figsize
        self.save = save

    def visualize(self, policy_result_dict, series_name, experiment_name, slice_result=None):
        """

        :param policy_result_dict:
        :param series_name:
        :param experiment_name:
        :param slice_result:
        :return:
        """
        df_list = list()
        extracted_index = None

        for policy_name, rep_result_dict_list in policy_result_dict.items():
            for rep_result_dict in rep_result_dict_list:
                seed_dict = rep_result_dict["seed"]

                action_df = rep_result_dict["action_experiment_df"]
                if extracted_index is None:
                    extracted_index = action_df["datetime"]
                action_df["policy_name"] = policy_name
                action_df["seed_mod"] = seed_dict["mod"]
                action_df["seed_pol"] = seed_dict["pol"]
                df_list.append(action_df)

        final_df = pd.concat(df_list, axis=0, ignore_index=True)  # type: pd.DataFrame

        if series_name == "conversion_rate":
            simulator_series_name = "cvr"
        else:
            simulator_series_name = series_name

        if "simulator.{}".format(simulator_series_name) not in final_df.columns:
            raise ValueError("Incorrect value of the series. There isn't series {} in experiment result DataFrame".format(simulator_series_name))

        for policy_name, group_df in final_df.groupby(["policy_name"]):
            policy_series_list = list()
            series_names_list = list()

            for seed_names, policy_group in group_df.groupby(['seed_mod', 'seed_pol'])["simulator.{}".format(simulator_series_name)]:
                series_names_list.append(seed_names)
                policy_series_list.append(policy_group.reset_index(drop=True))

            spec_dir = os.path.join("roomsage_simulator_starter", "experiment_specs", "historical")
            file_data_handler = FileDataHandler()
            experiment_spec = file_data_handler.load_spec(spec_dir,
                                                          "{}".format(experiment_name))
            dataset_name = experiment_spec["dataset_name"]
            file_data_handler = FileDataHandler()

            prior_df = file_data_handler.load_data("", dataset_name)
            prior_df["datetime"] = pd.to_datetime(prior_df["date"])
            prior_df = prior_df.set_index(prior_df["datetime"])
            prior_df = prior_df.loc[extracted_index[0]:, :]

            series_names_list.append("PRIOR")
            policy_series_list.append(prior_df[[series_name]].reset_index(drop=True))

            result_df = pd.concat(policy_series_list, axis=1)

            result_df = result_df.set_index(extracted_index)  # type: pd.DataFrame
            result_df = result_df.resample(rule="1M").sum()
            result_df.columns = series_names_list

            if slice_result is not None:
                result_df = result_df.loc[slice_result["start"]:slice_result["end"], :]

            ax = result_df.plot(figsize=self.figsize, alpha=0.5)

            ax.set_ylabel(series_name.capitalize().replace("_", " "))
            ax.set_title("Experiment results for {}, using {} policy".format(series_name, policy_name))
            # plt.tight_layout()
            x0, x1, y0, y1 = plt.axis()
            plot_margin = 2.25
            plt.axis((x0 - plot_margin,
                      x1 + plot_margin,
                      y0 - plot_margin,
                      y1 + plot_margin))

            if self.save:
                file_name = "experiment_repetition_results_[{}].png".format(series_name)
                plt.savefig(os.path.join(self.root_directory, experiment_name, file_name), bbox_inches="tight")
                plt.close()
            else:
                plt.show()


class RepResultCumSeriesPlotVisualizer(PlotVisualizer):

    def __init__(self, root_directory, figsize=(15, 7), save=False):
        PlotVisualizer.__init__(self, root_directory)
        self.figsize = figsize
        self.save = save

    def visualize(self, policy_result_dict, series_name, experiment_name, slice_result=None):
        """

        :param policy_result_dict:
        :param series_name:
        :param experiment_name:
        :param slice_result:
        :return:
        """
        df_list = list()
        extracted_index = None

        for policy_name, rep_result_dict_list in policy_result_dict.items():
            for rep_result_dict in rep_result_dict_list:
                seed_dict = rep_result_dict["seed"]

                action_df = rep_result_dict["action_experiment_df"]
                if extracted_index is None:
                    extracted_index = action_df["datetime"]
                action_df["policy_name"] = policy_name
                action_df["seed_mod"] = seed_dict["mod"]
                action_df["seed_pol"] = seed_dict["pol"]
                df_list.append(action_df)

        final_df = pd.concat(df_list, axis=0, ignore_index=True)  # type: pd.DataFrame

        if series_name not in final_df.columns:
            raise ValueError("Incorrect value of the series. There isn't series {} in experiment result DataFrame".format(series_name))

        for group_idx, group_df in final_df.groupby(['seed_mod', 'seed_pol']):
            policy_series_list = list()
            policy_name_list = list()

            for policy_name, policy_group in group_df.groupby(["policy_name"])[series_name]:
                policy_name_list.append(policy_name)
                policy_series_list.append(policy_group.reset_index(drop=True))

            result_df = pd.concat(policy_series_list, axis=1)

            result_df = result_df.set_index(extracted_index)  # type: pd.DataFrame
            result_df.columns = policy_name_list

            if slice_result is not None:
                result_df = result_df.loc[slice_result["start"]:slice_result["end"], :]

            ax = result_df.cumsum().plot(figsize=self.figsize, alpha=0.5)

            ax.set_ylabel(series_name.capitalize().replace("_", " "))
            ax.set_title("Experiment repetition results for cumulative sum of {}".format(series_name))

            if self.save:
                file_name = "experiment_repetition_results_[{}].png".format(series_name)
                plt.savefig(os.path.join(self.root_directory, experiment_name, file_name), bbox_inches="tight")
                plt.close()
            else:
                plt.show()


class SeriesPlotVisualizer(PlotVisualizer):

    def __init__(self, root_directory, chart_directory, experiment_name, state_type):
        super().__init__(root_directory, chart_directory, experiment_name)
        self.state_type = state_type

    def visualize(self, result_dict, save=False):
        plt.style.use("seaborn-whitegrid")

        action_policy_mean_results_df = result_dict['action_policy_mean_df']
        action_policy_std_results_df = result_dict['action_policy_std_df']
        action_policy_counter_df = result_dict['action_policy_counter_df']

        self.generate_cumulative_and_avg_series_plots(mean_df=action_policy_mean_results_df,
                                                      std_df=action_policy_std_results_df,
                                                      counter_df=action_policy_counter_df,
                                                      save=save)

    def generate_avg_series_plot(self, mean_df, std_df, counter_df, save=False):
        mean_series = mean_df / counter_df
        std_series = std_df / np.sqrt(counter_df)

        self._generate_plot_series(mean_df=mean_series, std_df=std_series, plot_type="avg", save=save)

    def generate_cumulative_series_plot(self, mean_df, std_df, save=False):
        self._generate_plot_series(mean_df=mean_df, std_df=std_df, plot_type="cum", save=save)

    def generate_cumulative_and_avg_series_plots(self, mean_df, std_df, counter_df, save=False):
        self.generate_cumulative_series_plot(mean_df=mean_df, std_df=std_df, save=save)
        self.generate_avg_series_plot(mean_df=mean_df, std_df=std_df, counter_df=counter_df, save=save)

    def _generate_plot_series(self, mean_df, std_df, plot_type="avg", save=False):

        if plot_type == "avg":
            y_label = "Average {} profit".format(self.state_type)
            title = "Average {} Profit Between Checkpoints".format(self.state_type.capitalize())
            file_name = "Chart_2_{}.png"
        elif plot_type == "cum":
            y_label = "Average cumulative profit"
            title = "Cumulative Profit Between Checkpoints"
            file_name = "Chart_1_{}.png"
        else:
            raise ValueError("Incorrect value of the plot_type")

        ax = mean_df.plot(figsize=(20, 7), marker='o', markersize=5, yerr=std_df)
        ax.set_xlabel("Day")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.legend(bbox_to_anchor=(0.5, -0.17), loc="upper center", frameon=False, ncol=2, mode="expand")

        if save:
            file_name = file_name.format(y_label.replace(" ", "_"))
            file_path = os.path.join(self.chart_directory, file_name)
            ensure_dir(file_path)
            plt.savefig(file_path, bbox_inches="tight")
            plt.show()

            plt.close()
        else:
            plt.show()


class ProfitBarPlotVisualizer(PlotVisualizer):

    def __init__(self, root_directory, chart_directory, experiment_name, state_type):
        super().__init__(root_directory, chart_directory, experiment_name)
        self.state_type = state_type
        plt.style.use("seaborn-whitegrid")

    def visualize(self, result_dict, save=False):

        action_policy_mean_results_df = result_dict['action_policy_mean_df']
        action_policy_counter_df = result_dict['action_policy_counter_df']

        self.generate_total_and_avg_bar_plots(mean_df=action_policy_mean_results_df,
                                              counter_df=action_policy_counter_df,
                                              save=save)

    def generate_total_and_avg_bar_plots(self, mean_df, counter_df, save=False):

        self.generate_total_bar_plot(mean_df, save)
        self.generate_average_bar_plot(mean_df, counter_df, save)

    def generate_total_bar_plot(self, mean_df, save=False):
        # Calculate total profit
        total_profit = mean_df.sum(axis=0)
        self._generate_bar_plot(series=total_profit, plot_type="total", save=save)

    def generate_average_bar_plot(self, mean_df, counter_df, save=False):
        # Calculate average profit
        avg_profit = mean_df.sum(axis=0) / counter_df.astype(float).sum(axis=0)

        result_str = ""
        for policy_name, avg_value in avg_profit.iteritems():
            result_str += "Policy: {}\n".format(policy_name)

            result_str += "Total profit={}\n".format(
                avg_value * counter_df.astype(float).sum(axis=0).loc[policy_name])
            result_str += "Average hourly profit={}\n".format(avg_value)
            result_str += "\n"

        print(result_str)

        self._generate_bar_plot(series=avg_profit, plot_type="avg", save=save)

    def _generate_bar_plot(self, series, plot_type="avg", save=False):

        max_result = np.max(np.max(series), 0)
        min_result = np.min(np.min(series), 0)
        results_range = np.max([max_result - min_result, 0.1])

        ax = series.plot(kind='bar', figsize=(20, 7))
        for p in ax.patches:
            shift = 0.04 * results_range if p.get_height() >= 0 else -0.1 * results_range
            ax.annotate("{}".format(np.round(p.get_height(), 2)),
                        (p.get_x() + p.get_width() / 2.0, p.get_height() + shift),
                        ha="center",
                        va="bottom")

        if plot_type == "avg":
            y_label = "Average {} profit".format(self.state_type)
            file_name = "Chart_4_{}.png"
        elif plot_type == "total":
            y_label = "Total profit"
            file_name = "Chart_3_{}.png"
        else:
            raise ValueError("Incorrect value of the plot_type")

        ax.set_ylabel(y_label)
        ax.set_title("{}s".format(y_label))

        if save:
            file_name = file_name.format(y_label.replace(" ", "_"))
            file_path = os.path.join(self.chart_directory, file_name)
            ensure_dir(file_path)
            plt.savefig(file_path, bbox_inches="tight")
            plt.show()

            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    viz = RepResultExperimentPlotVisualizer(os.path.join("data", "output", "Historical_Calibration"), figsize=(15, 5), save=False)

    # viz.generate_series_profit_rep_result("gadw_campagne_chessy_fr_hotel_spa_const_log_log_curve_const_cvr_is_1_181655974_42054076756_20170101_20181231")
    viz.generate_comparison_series_auctions_rep_result("Hotel_IT_OM_Hotel_IT_const_log_log_curve_real_cvr_is_1_20170101_20181231")
    viz.generate_comparison_series_clicks_rep_result("Hotel_IT_OM_Hotel_IT_const_log_log_curve_real_cvr_is_1_20170101_20181231")
    viz.generate_comparison_series_conversion_rate_rep_result("Hotel_IT_OM_Hotel_IT_const_log_log_curve_real_cvr_is_1_20170101_20181231")

    # viz.generate_series_bid_rep_result("Hotel_IT_OM_Hotel_IT_const_log_log_curve_real_cvr_is_1_20170101_20181231")
