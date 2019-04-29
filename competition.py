
# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from typing import List

import sys
import os

from copy import deepcopy

import numpy as np
import pandas as pd

from itertools import product
from datetime import datetime
from timeit import default_timer as timer

from ssa_sim_v2.tools.file_data_handler import FileDataHandler

from ssa_sim_v2.simulator.state import StateSet
from ssa_sim_v2.simulator.action import ActionSet, Action
from ssa_sim_v2.simulator.attribute import AttrSet
from ssa_sim_v2.simulator.competitive_date_how_simulator import CompetitiveDateHowSimulator
from ssa_sim_v2.result_aggregators.experiment_result_aggregator import ExperimentResultAggregator
from ssa_sim_v2.result_aggregators.experiment_plot_visualizer import SummaryExperimentPlotVisualizer
from ssa_sim_v2.result_aggregators.experiment_history_saver import ExperimentHistorySaver

from ssa_sim_v2.simulator.modules.auctions.auctions_base_module import AuctionsPoissonModule
from ssa_sim_v2.simulator.modules.auction_attributes.auction_attributes_base_module import AuctionAttributesModule
from ssa_sim_v2.simulator.modules.vickrey_auction.vickrey_auction_module import VickreyAuctionModule
from ssa_sim_v2.simulator.modules.competitive_click_probability.competitive_click_probability_base_module import CompetitiveClickProbabilityTwoClassGeometricModule
from ssa_sim_v2.simulator.modules.competitive_clicks.competitive_clicks_base_module import CompetitiveClicksBinomialModule
from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_base_module import ConversionRateFlatModule
from ssa_sim_v2.simulator.modules.conversions.conversions_base_module import ConversionsBinomialModule
from ssa_sim_v2.simulator.modules.revenue.revenue_base_module import RevenueGammaNoiseModule
from ssa_sim_v2.simulator.modules.competitive_cpc.competitive_cpc_base_module import CompetitiveCPCVickreyModule

from ssa_sim_v2.simulator.modules.auctions.auctions_date_how_module import AuctionsDateHoWModule
from ssa_sim_v2.simulator.modules.auction_attributes.auction_attributes_date_how_module import AuctionAttributesDateHoWModule
from ssa_sim_v2.simulator.modules.vickrey_auction.vickrey_auction_date_how_module import VickreyAuctionDateHoWModule
from ssa_sim_v2.simulator.modules.competitive_click_probability.competitive_click_probability_date_how_module import CompetitiveClickProbabilityDateHoWModule
from ssa_sim_v2.simulator.modules.competitive_clicks.competitive_clicks_date_how_module import CompetitiveClicksDateHoWModule
from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_date_how_module import ConversionRateDateHoWModule
from ssa_sim_v2.simulator.modules.conversions.conversions_date_how_module import ConversionsDateHoWModule
from ssa_sim_v2.simulator.modules.revenue.revenue_date_how_module import RevenueDateHoWModule
from ssa_sim_v2.simulator.modules.competitive_cpc.competitive_cpc_date_how_module import CompetitiveCpcDateHoWModule

from ssa_sim_v2.policies.policy import Policy
# noinspection PyUnresolvedReferences
# ------------------------------------------------------------

# load policies in root

from policy2019 import Policy2019



class Competition(object):
    """
    Base competition class.
    """

    def __init__(self):
        self.settings_spec_name = "settings"

    def run_competition(self):
        pass

    def _load_settings(self):
        pass

    def _load_experiment_specs(self, settings_spec):
        pass

    def _load_policies(self, settings_spec):
        pass

    def _run_simulation(self, batch, policy_infos, competition_start_dt):
        pass

    def _initialize_simulator(self, experiment_spec, seed):
        pass

    def _initialize_policies(self, policies, experiment_spec, seed):
        pass

    def _store_simulation_results(self):
        pass

    def _store_competition_results(self):
        pass


class SimpleCompetition(Competition):
    """
    Competition class performing a predefined number of runs on a stationary
    flat scenario for a given set of policies.
    """

    def __init__(self):
        super().__init__()
        self.save_history = True
        self.print_simulator_output = False
        self.processing = "parallel"
        self.num_cores = 4
        self.master_seed = 4444
        self.n_experiment_reps = 3

        self.date_from = None
        self.date_to = None
        self.time_steps_df = pd.DataFrame()
        self.dates_list = []

        self.simulator = None  # type: CompetitiveDateHowSimulator
        self.experiment_specs = []
        self.policy_infos = []

        self.state_set = None
        self.action_set = None
        self.attr_set = None

    def run_competition(self):
        competition_start_dt = datetime.now().strftime("%Y%m%d_%H%M")
        file_data_handler = FileDataHandler()
        root_directory = file_data_handler.get_output_full_path(os.path.join(competition_start_dt))

        settings_spec = self._load_settings()
        experiment_specs = self._load_experiment_specs(settings_spec)
        policy_infos = self._load_policies(settings_spec)

        # Initialize seeds

        seed_min = 100000
        seed_max = 999999

        rng = np.random.RandomState(seed=self.master_seed)

        seeds = [{}] * self.n_experiment_reps
        for rep in range(self.n_experiment_reps):
            seeds[rep] = {"mod": rng.randint(low=seed_min, high=seed_max),
                          "pol": rng.randint(low=seed_min, high=seed_max)}

        # Define simulations

        batches = list(product(seeds, experiment_specs))

        # Run simulations

        if self.processing == "serial":

            # Run batch serially
            start_time = timer()
            res_ser = []
            print("Starting to run {} reps of {} simulations with {} time steps serially".format(
                self.n_experiment_reps, len(experiment_specs), len(self.time_steps_df)))
            # Run batch
            for b in batches:
                temp = self._run_simulation(b, policy_infos, competition_start_dt)

                res_ser.append(temp)
            results = res_ser

        elif self.processing == "parallel":

            # Run batch in parallel
            import pathos.multiprocessing as mp

            pool = mp.Pool(processes=self.num_cores)
            start_time = timer()
            print("Starting to run {} reps of {} simulations with {} time steps in parallel".format(
                self.n_experiment_reps, len(experiment_specs), len(self.time_steps_df)))
            res_par = pool.starmap(self._run_simulation,
                                   [(b, policy_infos, competition_start_dt) for b in batches])
            pool.close()
            pool.join()
            results = res_par

        print("Done running {} reps of {} simulations with {} time steps in {:.2f} minutes".format(
            self.n_experiment_reps, len(experiment_specs), len(self.time_steps_df),
            (timer() - start_time) / 60))

        aggregator = ExperimentResultAggregator(root_directory)
        processed_result = aggregator.aggregate_resampled_experiment_result(result_list=results, rule="1D",
                                                                            policy_order=[p["name"] for p in policy_infos],
                                                                            agg_save=True)

        visualizer = SummaryExperimentPlotVisualizer(root_directory=os.path.join(root_directory, "resampled_1D"),
                                                     chart_directory=os.path.join(root_directory, "charts"),
                                                     experiment_name="competition",
                                                     state_type="hourly")
        visualizer.generate_summary_profit_bar_plots(processed_result, save=True)
        visualizer.generate_summary_profit_series_plots(processed_result, save=True)

    def _load_settings(self):
        file_data_handler = FileDataHandler()
        settings_spec = file_data_handler.load_spec(os.path.join(""),
                                                    self.settings_spec_name)

        if settings_spec is None:
            sys.exit("Can't load experiment spec")

        self.save_history = settings_spec["save_history"]
        self.print_simulator_output = settings_spec["print_simulator_output"]
        self.processing = settings_spec["processing"]
        self.num_cores = settings_spec["num_cores"]
        self.master_seed = settings_spec["master_seed"]
        self.n_experiment_reps = settings_spec["n_experiment_reps"]

        # Initialize state set
        self.date_from = settings_spec.get("date_from", None)
        self.date_to = settings_spec.get("date_to", None)

        self.time_steps_df, self.dates_list = \
            self._get_time_steps_data(self.date_from, self.date_to)

        self.state_set = StateSet(["date", "how"], ["discrete", "discrete"],
                                  [self.dates_list, list(range(168))])

        # Initialize attribute set
        names = settings_spec["attributes"]["names"]
        vals = settings_spec["attributes"]["vals"]
        self.attr_set = AttrSet(names, vals)

        # Initialize action set
        self.action_set = ActionSet(self.attr_set, max_bid=9.99, min_bid=0.01, max_mod=9.0, min_mod=0.1)

        return settings_spec

    def _load_experiment_specs(self, settings_spec):
        experiment_spec_names = settings_spec["experiment_specs"]
        file_data_handler = FileDataHandler()
        for experiment_spec_name in experiment_spec_names:
            experiment_spec = file_data_handler.load_spec(os.path.join("ssa_sim_v2",
                                                                       "experiment_specs"),
                                                          "{}".format(experiment_spec_name))

            if experiment_spec is None:
                sys.exit("Can't load experiment spec")

            self.experiment_specs.append({"name": experiment_spec_name,
                                          "spec": experiment_spec})

        return self.experiment_specs

    def _load_policies(self, settings_spec):
        self.policy_infos = settings_spec["policies"]
        return self.policy_infos

    def _run_simulation(self, batch, policy_infos, competition_start_dt):
        seed, experiment_spec = batch
        experiment_spec_name = experiment_spec["name"]
        experiment_spec = experiment_spec["spec"]

        file_data_handler = FileDataHandler()
        root_directory = file_data_handler.get_output_full_path(
            os.path.join(competition_start_dt, experiment_spec_name))

        simulator = self._initialize_simulator(experiment_spec, seed["mod"])
        policies = self._initialize_policies(policy_infos, experiment_spec, seed["pol"])

        saver = ExperimentHistorySaver(root_directory=root_directory,
                                       seed=seed,
                                       experiment_spec=experiment_spec,
                                       policy_infos=policy_infos)

        state = deepcopy(simulator.state)

        while state is not None:

            # Act

            actions = []  # type: List[Action]
            for policy in policies:
                original_state = deepcopy(simulator.state)
                actions.append(policy.act(original_state))
            original_state = deepcopy(simulator.state)

            # Simulator step

            step_results = simulator.step(actions)
            state = simulator.state

            # Learn

            for i in range(len(policies)):
                original_state_copy = deepcopy(original_state)
                policies[i].learn(original_state_copy, step_results[i])

            # Store results

            for i in range(len(policies)):
                saver.save_experiment_iteration(
                    policy_name=policy_infos[i]["name"],
                    policy=policies[i],
                    profit=step_results[i]["info"]["profit_is"],
                    state=original_state.values,
                    action=step_results[i]["action"],
                    effective_action=step_results[i]["effective_action"],
                    info=step_results[i]["info"],
                    attr_info=step_results[i]["attr_info"])

        results = saver.generate_experiment_results(save=self.save_history, csv=True)

        return results

    def _initialize_simulator(self, experiment_spec, seed):
        def initialize_priors(params, base_class):
            attr_combinations = list(self.attr_set.get_all_attr_tuples())
            priors = self.time_steps_df.copy()
            priors.loc[:, "prior"] = pd.Series([dict.fromkeys(attr_combinations, params)]*len(priors))

            base_classes = self.time_steps_df.copy()
            base_classes.loc[:, "base_class"] = base_class

            return priors, base_classes

        # Initialize auctions priors
        module_class = AuctionsPoissonModule
        Params = module_class.Params
        params = Params(auctions=1000)
        priors = self.time_steps_df.copy()
        priors.loc[:, "prior"] = [{(): params}]*len(priors)
        base_classes = self.time_steps_df.copy()
        base_classes.loc[:, "base_class"] = module_class
        auctions_priors = priors
        auctions_base_classes = base_classes

        # Initialize auction_attributes priors
        module_class = AuctionAttributesModule
        Params = module_class.Params
        params = Params(p=1.0)  # Probabilities are normalized
        auction_attributes_priors, auction_attributes_base_classes \
            = initialize_priors(params, module_class)

        # Initialize vickrey_auction priors
        module_class = VickreyAuctionModule
        Params = module_class.Params
        params = Params()
        vickrey_auction_priors, vickrey_auction_base_classes \
            = initialize_priors(params, module_class)

        # Initialize competitive_click_probability priors
        module_class = CompetitiveClickProbabilityTwoClassGeometricModule
        Params = module_class.Params
        params = Params(n_pos=8, p=0.5, q=0.5, r_11=0.6, r_12=0.4, r_2=0.5)
        competitive_click_probability_priors, competitive_click_probability_base_classes \
            = initialize_priors(params, module_class)

        # Initialize competitive_clicks priors
        module_class = CompetitiveClicksBinomialModule
        Params = module_class.Params
        params = Params(noise_level=0.0, noise_type="multiplicative")
        competitive_clicks_priors, competitive_clicks_base_classes \
            = initialize_priors(params, module_class)

        # Initialize conversion_rate priors
        module_class = ConversionRateFlatModule
        Params = module_class.Params
        params = Params(cvr=0.02, noise_level=0.0, noise_type="multiplicative")
        conversion_rate_priors, conversion_rate_base_classes \
            = initialize_priors(params, module_class)

        # Initialize conversions priors
        module_class = ConversionsBinomialModule
        Params = module_class.Params
        params = Params(noise_level=0.0, noise_type="multiplicative")
        conversions_priors, conversions_base_classes \
            = initialize_priors(params, module_class)

        # Initialize revenue priors
        module_class = RevenueGammaNoiseModule
        Params = module_class.Params
        params = Params(avg_rpv=300.0, noise_level=100.0)
        revenue_priors, revenue_base_classes = initialize_priors(
            params, module_class)

        # Initialize competitive_cpc priors
        module_class = CompetitiveCPCVickreyModule
        Params = module_class.Params
        params = Params(n_pos=8, fee=0.01)
        competitive_cpc_priors, competitive_cpc_base_classes = \
            initialize_priors(params, module_class)

        # Module setup for the simulator
        modules = \
            {"auctions": AuctionsDateHoWModule(auctions_priors,
                                               auctions_base_classes,
                                               seed),
             "auction_attributes": AuctionAttributesDateHoWModule(auction_attributes_priors,
                                                                  auction_attributes_base_classes,
                                                                  seed),
             "vickrey_auction": VickreyAuctionDateHoWModule(vickrey_auction_priors,
                                                            vickrey_auction_base_classes,
                                                            seed),
             "competitive_click_probability": CompetitiveClickProbabilityDateHoWModule(competitive_click_probability_priors,
                                                                                       competitive_click_probability_base_classes,
                                                                                       seed),
             "competitive_clicks": CompetitiveClicksDateHoWModule(competitive_clicks_priors,
                                                                  competitive_clicks_base_classes,
                                                                  seed),
             "conversion_rate": ConversionRateDateHoWModule(conversion_rate_priors,
                                                            conversion_rate_base_classes,
                                                            seed),
             "conversions": ConversionsDateHoWModule(conversions_priors,
                                                     conversions_base_classes,
                                                     seed),
             "revenue": RevenueDateHoWModule(revenue_priors,
                                             revenue_base_classes,
                                             seed),
             "competitive_cpc": CompetitiveCpcDateHoWModule(competitive_cpc_priors,
                                                            competitive_cpc_base_classes,
                                                            seed)
             }

        simulator = CompetitiveDateHowSimulator(self.state_set, self.action_set, self.attr_set,
                                                modules, self.date_from, self.date_to,
                                                income_share=experiment_spec.get("policy_defaults",
                                                                                 {"udp": {"income_share": 1.0}})
                                                .get("udp", {"income_share": 1.0}).get("income_share", 1.0),
                                                print_trace=self.print_simulator_output)

        return simulator

    def _initialize_policies(self, policy_infos, experiment_spec, seed):
        param_defaults = experiment_spec.get("policy_defaults", {})
        policies = []
        for policy_info in policy_infos:
            params = deepcopy(param_defaults)
            params.update(policy_info["params"])
            policy_class = globals()[policy_info["policy"]]
            if "seed" in params.keys():
                this_seed = params["seed"]
            else:
                this_seed = seed
            policy = policy_class(self.state_set,
                                  self.action_set,
                                  self.attr_set,
                                  seed=this_seed,
                                  save_history=self.save_history)  # type: Policy
            policy.initialize(params)
            policies.append(policy)

        return policies

    def _store_simulation_results(self):
        pass

    def _store_competition_results(self):
        pass

    def _get_time_steps_data(self, date_from, date_to):
        tmp_df = pd.DataFrame(np.array(range(24)), columns=["hour_of_day"])
        tmp_df["key"] = 1
        time_steps_df = pd.DataFrame(pd.date_range(date_from, date_to), columns=["date"])
        dates_list = time_steps_df["date"].tolist()
        time_steps_df["key"] = 1
        time_steps_df = pd.merge(time_steps_df, tmp_df, on=["key"], how="left")  # columns: ['date', 'hour_of_day']
        time_steps_df["hour_of_week"] = pd.to_datetime(time_steps_df["date"]).dt.dayofweek * 24 + time_steps_df["hour_of_day"]
        time_steps_df["date"] = time_steps_df["date"].dt.strftime("%Y-%m-%d")
        time_steps_df = time_steps_df[["date", "hour_of_week"]]
        return time_steps_df, dates_list


if __name__ == "__main__":
    from policy_thompson import PolicyThompsonSamplingSI   # this line is needed. If your policy class is in somewhere else, please import accordingly
    # if your policy is from filename.py with classname Policy_My_Bid, the import is done as:
    # from filename import Policy_My_Bid

    competition = SimpleCompetition()
    competition.run_competition()
    # Most of competition elements can be changed by editing settings.spec
    # "policies": [
    #     {
    #         "name": "Player",
    #         "policy": "PolicyThompsonSamplingSI",
    #         "params": {"seed": 9090}
    #     },
    #     {
    #         "name": "Random 1",
    #         "policy": "Policy2019",
    #         "params": {"seed": 1234}
    #     },
    #     {
    #         "name": "Random 2",
    #         "policy": "Policy2019",
    #         "params": {"seed": 64323}
    #     },
    #     {
    #         "name": "Random 3",
    #         "policy": "Policy2019",
    #         "params": {"seed": 418}
    #     },
    #     {
    #         "name": "Random 4",
    #         "policy": "Policy2019",
    #         "params": {"seed": 2019}
    #     },
    #     {
    #         "name": "Random 5",
    #         "policy": "Policy2019",
    #         "params": {"seed": 1234567}
    #     },
    #     {
    #         "name": "Random 6",
    #         "policy": "Policy2019",
    #         "params": {"seed": 2323}
    #     },
    #     {
    #         "name": "Random 7",
    #         "policy": "Policy2019",
    #         "params": {"seed": 98765}
    #     }
    # ],
