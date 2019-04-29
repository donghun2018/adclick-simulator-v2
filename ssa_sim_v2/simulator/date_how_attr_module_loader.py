# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------
    
import pandas as pd

from ssa_sim_v2.simulator.module_loader import ModuleLoader

from ssa_sim_v2.simulator.modules.multi_state_simulator_module import MultiStateSimulatorModule
from ssa_sim_v2.prior_generators.base.data_base.historical_prior_generator import HistoricalPriorGenerator
from ssa_sim_v2.prior_generators.base.data_base.synthetic_prior_generator import SyntheticPriorGenerator

# noinspection PyUnresolvedReferences
from ssa_sim_v2.prior_generators.generators.auctions.date_how import *
# noinspection PyUnresolvedReferences
from ssa_sim_v2.prior_generators.generators.auction_attributes.date_how import *
# noinspection PyUnresolvedReferences
from ssa_sim_v2.prior_generators.generators.average_position.date_how import *
# noinspection PyUnresolvedReferences
from ssa_sim_v2.prior_generators.generators.click_probability.date_how import *
# noinspection PyUnresolvedReferences
from ssa_sim_v2.prior_generators.generators.clicks.date_how import *
# noinspection PyUnresolvedReferences
from ssa_sim_v2.prior_generators.generators.conversion_rate.date_how import *
# noinspection PyUnresolvedReferences
from ssa_sim_v2.prior_generators.generators.conversions.date_how import *
# noinspection PyUnresolvedReferences
from ssa_sim_v2.prior_generators.generators.cpc.date_how import *
# noinspection PyUnresolvedReferences
from ssa_sim_v2.prior_generators.generators.revenue.date_how import *

# ------------------------------------------------------------


class DateHoWAttrModuleLoader(ModuleLoader):
    """
    Module loader providing load_modules method which returns a dictionary
    with date-hour of week modules with attributes initialized from provided
    real priors or prior generators according to an experiment spec.
    """

    def __init__(self):
        ModuleLoader.__init__(self)

    def load_modules(self, experiment_spec, data, seed=12345):
        """
        Loads date-hour of week modules according to the experiment spec.

        :param dict experiment_spec: Experiment specification.
        :param pd.DataFrame data: Input data.
        :param int seed: Seed for the random number generator.
        :return: action_set dictionary with initialized modules to be used by a simulator.
        :rtype: dict
        """

        assert experiment_spec, self.experiment_spec_assert_message

        date_from = experiment_spec["date_from"]
        date_to = experiment_spec["date_to"]
        modules_request = experiment_spec["modules_request"]

        assert not data.empty, self.data_input_assert_message

        # Prepare modules
        self.modules = {}

        # Simple DataFrame preprocessing
        # Add "hour_of_week" column if it does not exist in the data
        if (type(data) == pd.DataFrame) and ("hour_of_week" not in data.columns) and ("hour_of_day" in data.columns):
            data["hour_of_week"] = pd.to_datetime(data["date"]).dt.dayofweek * 24 + data["hour_of_day"]

        params = {}

        if "auctions" in modules_request.keys():
            self.modules["auctions"] = self.load_module(
                var_name="auctions", params=params, data=data,
                modules_request=modules_request,
                date_from=date_from, date_to=date_to, seed=seed)

        if "auction_attributes" in modules_request.keys():
            self.modules["auction_attributes"] = self.load_module(
                var_name="auctions", params=params, data=data,
                modules_request=modules_request,
                date_from=date_from, date_to=date_to, seed=seed)

            params.update({"attribute_priors": self.modules["auction_attributes"].priors})

        if "click_probability" in modules_request.keys():
            self.modules["click_probability"] = self.load_module(
                var_name="click_probability", params=params, data=data,
                modules_request=modules_request,
                date_from=date_from, date_to=date_to, seed=seed)

        if "clicks" in modules_request.keys():
            self.modules["clicks"] = self.load_module(
                var_name="clicks", params=params, data=data,
                modules_request=modules_request,
                date_from=date_from, date_to=date_to, seed=seed)

        if "conversion_rate" in modules_request.keys():
            self.modules["conversion_rate"] = self.load_module(
                var_name="conversion_rate", params=params, data=data,
                modules_request=modules_request,
                date_from=date_from, date_to=date_to, seed=seed)

        if "conversions" in modules_request.keys():
            self.modules["conversions"] = self.load_module(
                var_name="conversions", params=params, data=data,
                modules_request=modules_request,
                date_from=date_from, date_to=date_to, seed=seed)

        if "cpc" in modules_request.keys():
            self.modules["cpc"] = self.load_module(
                var_name="cpc", params=params, data=data,
                modules_request=modules_request,
                date_from=date_from, date_to=date_to, seed=seed)

        if "revenue" in modules_request.keys():
            self.modules["revenue"] = self.load_module(
                var_name="revenue", params=params, data=data,
                modules_request=modules_request,
                date_from=date_from, date_to=date_to, seed=seed)

        if "average_position" in modules_request.keys():
            self.modules["average_position"] = self.load_module(
                var_name="average_position", params=params, data=data,
                modules_request=modules_request,
                date_from=date_from, date_to=date_to, seed=seed)

        if "avg_price" in modules_request.keys():
            self.modules["avg_price"] = self.load_module(
                var_name="avg_price", params=params, data=data,
                modules_request=modules_request,
                date_from=date_from, date_to=date_to, seed=seed)

        return self.modules

    def load_module(self, var_name, params, data, modules_request, date_from, date_to, seed):
        """
        Loads an appropriate module for the variable var_name using an appropriate
        prior generator defined in modules_request.

        :param str var_name: The variable name for which a module is to be
            created. simulator.g. auctions, cpc, conversions etc.
        :param dict params: A dictionary with potential additional params.
            for the prior generator
        :param pd.DataFrame data: A DataFrame with historical data.
        :param dict modules_request: A dictionary with a modules specification.
        :param str date_from: Starting date -- first state. Format "yyyy-mm-dd".
        :param str date_to: End date -- last state. Format "yyyy-mm-dd".
        :param int seed: Seed for the random number generator.
        :return: An appropriate module for the variable var_name using an appropriate
            prior generator defined in modules_request.
        :rtype: MultiStateSimulatorModule
        """

        if isinstance(modules_request[var_name]["class"], HistoricalPriorGenerator):
            return self.load_module_based_on_historical_data(
                var_name=var_name,
                params=params,
                data=data,
                modules_request=modules_request,
                date_from=date_from,
                date_to=date_to,
                seed=seed)
        elif isinstance(modules_request[var_name]["class"], SyntheticPriorGenerator):
            return self.load_module_based_on_synthetic_data(
                var_name=var_name,
                params=params,
                modules_request=modules_request,
                date_from=date_from,
                date_to=date_to,
                seed=seed)

    def load_module_based_on_historical_data(
            self, var_name, params, data, modules_request, date_from, date_to, seed):
        """
        Loads an appropriate module for the variable var_name using an appropriate
        prior generator defined in modules_request. Adds historical data to params
        before calling the prior generator.

        :param str var_name: The variable name for which a module is to be
            created. simulator.g. auctions, cpc, conversions etc.
        :param dict params: A dictionary with potential additional params.
            for the prior generator
        :param pd.DataFrame data: A DataFrame with historical data.
        :param dict modules_request: A dictionary with a modules specification.
        :param str date_from: Starting date -- first state. Format "yyyy-mm-dd".
        :param str date_to: End date -- last state. Format "yyyy-mm-dd".
        :param int seed: Seed for the random number generator.
        :return: An appropriate module for the variable var_name using an appropriate
            prior generator defined in modules_request.
        :rtype: MultiStateSimulatorModule
        """

        params = self.add_historical_data_to_params(params, var_name, data, modules_request)

        return self.initialize_module(var_name, params, modules_request, date_from, date_to, seed)

    def load_module_based_on_synthetic_data(
            self, var_name, params, modules_request, date_from, date_to, seed):
        """
        Loads an appropriate module for the variable var_name using an appropriate
        prior generator defined in modules_request. Assumes a completely synthetic
        prior generator.

        :param str var_name: The variable name for which a module is to be
            created. simulator.g. auctions, cpc, conversions etc.
        :param dict params: A dictionary with potential additional params.
            for the prior generator
        :param dict modules_request: A dictionary with a modules specification.
        :param str date_from: Starting date -- first state. Format "yyyy-mm-dd".
        :param str date_to: End date -- last state. Format "yyyy-mm-dd".
        :param int seed: Seed for the random number generator.
        :return: An appropriate module for the variable var_name using an appropriate
            prior generator defined in modules_request.
        :rtype: object
        """
        return self.initialize_module(var_name, params, modules_request, date_from, date_to, seed)

    def initialize_module(self, var_name, params, modules_request, date_from, date_to, seed):
        module_generator_constructor = globals()[modules_request[var_name]["class"]]
        module_generator = module_generator_constructor(seed=seed)

        if "params" in modules_request[var_name]:
            params.update(modules_request[var_name]["params"])

        loaded_module = module_generator.get_module(date_from=date_from,
                                                    date_to=date_to,
                                                    params=modules_request[var_name]["params"])

        return loaded_module

    def add_historical_data_to_params(
            self, params, var_name, data, modules_request):

        historical_priors = None

        if var_name == "auctions":
            historical_priors = self.prepare_auctions_prior_data(data, modules_request)
        elif var_name == "auction_attributes":
            historical_priors = self.prepare_auction_attributes_prior_data(data, modules_request)
        elif var_name == "click_probability":
            historical_priors = self.prepare_click_probability_prior_data(data, modules_request)
        elif var_name == "clicks":
            historical_priors = self.prepare_clicks_prior_data(data, modules_request)
        elif var_name == "conversion_rate":
            historical_priors = self.prepare_conversion_rate_prior_data(data, modules_request)
        elif var_name == "conversions":
            historical_priors = self.prepare_conversions_prior_data(data, modules_request)
        elif var_name == "cpc":
            historical_priors = self.prepare_cpc_prior_data(data, modules_request)
        elif var_name == "revenue":
            historical_priors = self.prepare_revenue_prior_data(data, modules_request)
        elif var_name == "average_position":
            historical_priors = self.prepare_ad_position_prior_data(data, modules_request)
        elif var_name == "avg_price":
            historical_priors = self.prepare_avg_price_prior_data(data, modules_request)

        params["historical_prior"] = historical_priors

        return params




    def load_average_position_module(self, data, modules_request, date_from, date_to, seed):
        average_position_module = None

        if modules_request["average_position"]["class"] == "AveragePositionHyperbolicDateHoW":
            assert "click_probability" in self.modules.keys(), "There is no click_probability column in [data] DataFrame"

            max_cp_df = data.loc[(data["date"] >= date_from)
                                 & (data["date"] <= date_to), ["date", "hour_of_week"]].reset_index(drop=True)

            max_cp_df["max_cp"] = pd.Series([self.modules["click_probability"].get_p(bid=float("inf"),
                                                                                     date=max_cp_df["date"][t],
                                                                                     how=max_cp_df["hour_of_week"][t])
                                             for t in range(len(max_cp_df))])

            average_position_module = AveragePositionHyperbolicDateHoW(max_cp=max_cp_df, seed=seed)

        return average_position_module

    @staticmethod
    def load_module_using_prior_generator(variable_name, modules_request, date_from,
                                          date_to, seed, class_key="class"):
        """
        Loads an appropriate module_loader for the variable variable_name using a prior
        from the synthetic module_loader generator defined in modules_request.

        :param str variable_name: The variable name for which a module_loader is to be
            created. simulator.g. avg_price, cpc, conversions etc.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.
        :param str class_key: The key used in the modules_request to indicate
            the class module_loader used.
        :return: An appropriate module for the variable variable_name using a prior
            from the synthetic module generator defined in modules_request.
        :rtype: object
        """

        module_generator_constructor = globals()[modules_request[variable_name][class_key]]
        module_generator = module_generator_constructor(seed=seed)

        if "params" in modules_request[variable_name]:
            loaded_module = module_generator.get_module(date_from=date_from,
                                                        date_to=date_to,
                                                        params=modules_request[variable_name]["params"])
        else:
            loaded_module = module_generator.get_module(date_from=date_from,
                                                        date_to=date_to,
                                                        params={})

        return loaded_module

    def load_auctions_module_with_prior(self, data, modules_request, seed):
        """
        Loads a date-how auctions module_loader as defined in modules_request based on
        the prior provided in data.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param int seed: Seed for the random number generator.
        :return: An auctions module_loader as defined in modules_request.
        :rtype: AuctionsDateHow
        """

        assert "auctions" in data.columns, "There is no auctions column in [data] DataFrame"

        auction_module = None

        if "AuctionsPoisson" in modules_request["auctions"]["class"]:
            df = self.prepare_auctions_prior_data(data=data, modules_request=modules_request)

            auction_module = AuctionsPoissonDateHoW(L=df, seed=seed)

        return auction_module

    def prepare_auctions_prior_data(self, data, modules_request):
        """
        Prepares a prior for the auctions module_loader.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :return: Prior for the auctions module_loader.
        :rtype: pd.DataFrame
        """

        auctions_df = None

        state_list = ["date", "hour_of_week"]

        if modules_request["auctions"]["auctions"] == "original":
            column_list = state_list.copy()
            column_list.extend(["auctions"])

            auctions_df = data.loc[:, column_list]

        elif modules_request["auctions"]["auctions"] == "smoothed":
            assert "auctions.smoothed" in data.columns, "There is no auctions.smoothed column in [data] DataFrame"

            column_list = state_list.copy()
            column_list.extend(["auctions.smoothed"])

            auctions_df = data.loc[:, column_list]
            auctions_df = auctions_df.rename(columns={"auctions.smoothed": "auctions"})

        elif modules_request["auctions"]["auctions"] == "decomposed":
            column_list = state_list.copy()
            column_list.extend(["auctions.smoothed"])

            auctions_df = data.loc[:, column_list]
            auctions_df["auctions"] = self.concatenate_decomposed_series(data=data,
                                                                         params=modules_request["auctions"]["params"])

        return auctions_df

    def load_avg_price_module_with_prior(self, data, modules_request, seed):
        """
        Loads a date-how average price module_loader as defined in modules_request
        based on the prior provided in data.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param int seed: Seed for the random number generator.
        :return: An average price module_loader as defined in modules_request.
        :rtype: AvgPriceDateHow
        """

        assert "avg_price" in data.columns, "There is no avg_price column in [data] DataFrame"

        avg_price_module = None

        if "AvgPriceHistoricalAvg" in modules_request["avg_price"]["class"]:
            df = self.prepare_avg_price_prior_data(data=data, modules_request=modules_request)

            avg_price_module = AvgPriceHistoricalAvgDateHoW(avg_price=df, seed=seed)

        return avg_price_module

    def prepare_avg_price_prior_data(self, data, modules_request):
        """
        Prepares a prior for the average price module_loader.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :return: Prior for the average price module_loader.
        :rtype: pd.DataFrame
        """

        avg_price_df = None

        state_list = ["date", "hour_of_week"]

        if modules_request["avg_price"]["avg_price"] == "original":
            column_list = state_list.copy()
            column_list.extend(["avg_price"])

            avg_price_df = data.loc[:, column_list]

        elif modules_request["avg_price"]["avg_price"] == "smoothed":
            assert "avg_price.smoothed" in data.columns, "There is no avg_price column in [data] DataFrame"
            column_list = state_list.copy()
            column_list.extend(["avg_price.smoothed"])

            avg_price_df = data.loc[:, column_list]
            avg_price_df = avg_price_df.rename(columns={"avg_price.smoothed": "avg_price"})

        elif modules_request["avg_price"]["avg_price"] == "decomposed":
            column_list = state_list.copy()
            column_list.extend(["avg_price"])

            avg_price_df = data.loc[:, column_list]
            avg_price_df["avg_price"] = self.concatenate_decomposed_series(data=data,
                                                                           params=modules_request["avg_price"]["params"])

        return avg_price_df

    def load_click_probability_module_with_prior(self, data, modules_request, seed):
        """
        Loads a date-how click probability module_loader as defined in modules_request
        based on the prior provided in data.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param int seed: Seed for the random number generator.
        :return: action_set click probability module_loader as defined in modules_request.
        :rtype: ClickProbabilityDateHow
        """

        assert "click_probability" in data.columns, "There is no click_probability column in [data] DataFrame"
        assert "weighted_bid" in data.columns, "There is no weighted_bid column in [data] DataFrame"

        click_prob_class = self.select_click_probability_class(modules_request=modules_request)

        click_probability_df = self.prepare_click_probability_prior_data(data=data,
                                                                         modules_request=modules_request)

        bid_df = self.prepare_weighted_bid_prior_data(data=data,
                                                      modules_request=modules_request,
                                                      variable_name="clicks")

        click_prob_module = click_prob_class(click_probability_df, bid_df, seed=seed)

        return click_prob_module

    @staticmethod
    def select_click_probability_class(modules_request):
        """
        Returns the click probability class according to the modules request.

        :param dict modules_request: action_set dictionary with a modules specification.
        :return: Click probability class according to the modules request.
        :rtype: class
        """

        click_prob_class = None

        if modules_request["clicks"]["cp_class"] == "ClickProbabilityLogisticDateHoW":
            click_prob_class = ClickProbabilityLogisticDateHoW
        elif modules_request["clicks"]["cp_class"] == "ClickProbabilityLogisticLogDateHoW":
            click_prob_class = ClickProbabilityLogisticLogDateHoW

        return click_prob_class

    def prepare_click_probability_prior_data(self, data, modules_request):
        """
        Prepare click probability prior for the click probability module_loader.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :return: Prior for the click probability module_loader.
        :rtype: pd.DataFrame
        """

        click_probability_df = None

        state_list = ["date", "hour_of_week"]

        if modules_request["clicks"]["click_probability"] == "original":
            column_list = state_list.copy()
            column_list.extend(["click_probability"])

            click_probability_df = data.loc[:, column_list]

        elif modules_request["clicks"]["click_probability"] == "smoothed":
            assert "click_probability.smoothed" in data.columns, \
                "There is no click_probability.smoothed column in [data] DataFrame"

            column_list = state_list.copy()
            column_list.extend(["click_probability.smoothed"])

            click_probability_df = data.loc[:, column_list]
            click_probability_df = click_probability_df.rename(columns={
                "click_probability.smoothed": "click_probability"
            })

        elif modules_request["clicks"]["click_probability"] == "decomposed":
            column_list = state_list.copy()
            column_list.extend(["click_probability"])

            click_probability_df = data.loc[:, column_list]
            click_probability_df["click_probability"] = self.concatenate_decomposed_series(
                                                data=data,
                                                params=modules_request["clicks"]["params"]["click_probability_params"])

        return click_probability_df

    def prepare_weighted_bid_prior_data(self, data, modules_request, variable_name):
        """
        Prepares weighted bid prior.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str variable_name: Variable name for the module_loader for which this
            prior is being prepared, e.g. "clicks" or "cpc".
        :return: Weigted bid prior.
        :rtype: pd.DataFrame
        """

        bid_df = None

        state_list = ["date", "hour_of_week"]

        if modules_request[variable_name]["weighted_bid"] == "original":
            column_list = state_list.copy()
            column_list.extend(["weighted_bid"])

            bid_df = data.loc[:, column_list]
            bid_df = bid_df.rename(columns={"weighted_bid": "cpc_bid"})

        elif modules_request[variable_name]["weighted_bid"] == "smoothed":
            assert "weighted_bid.smoothed" in data.columns, \
                "There is no weighted_bid.smoothed column in [data] DataFrame"

            column_list = state_list.copy()
            column_list.extend(["weighted_bid.smoothed"])

            bid_df = data.loc[:, column_list]
            bid_df = bid_df.rename(columns={"weighted_bid.smoothed": "cpc_bid"})

        elif modules_request[variable_name]["weighted_bid"] == "decomposed":
            column_list = state_list.copy()
            column_list.extend(["weighted_bid"])

            bid_df = data.loc[:, column_list]
            bid_df["weighted_bid"] = self.concatenate_decomposed_series(
                                                data=data,
                                                params=modules_request[variable_name]["params"]["weighted_bid_params"])
            bid_df = bid_df.rename(columns={"weighted_bid": "cpc_bid"})

        return bid_df

    def load_conversion_rate_module_with_prior(self, data, modules_request, seed):
        """
        Loads a date-how conversion rate module as defined in modules_request
        based on the prior provided in data.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param int seed: Seed for the random number generator.
        :return: action_set conversions module_loader as defined in modules_request.
        :rtype: ConversionsDateHow
        """

        assert "conversion_rate" in data.columns, "There is no conversion_rate column in [data] DataFrame"

        conversion_rate_module = None

        if "ConversionRateFlat" in modules_request["conversion_rate"]["class"]:
            df = self.prepare_conversion_rate_prior_data(data=data, modules_request=modules_request)

            #     elif modules_request["conversion_rate"]["class"] == "ConversionRateFlatDate":
            #
            #     assert data is not None, self.data_input_assert_message
            #     assert ("conversion_rate" in data.columns)
            #
            #     if modules_request["conversion_rate"]["conversion_rate"] == "original":
            #         conversion_rates = data.loc[:, ["date", "conversion_rate"]]
            #     elif modules_request["conversion_rate"]["conversion_rate"] == "smoothed":
            #         assert ("conversion_rate.smoothed" in data.columns)
            #         conversion_rates = data.loc[:, ["date", "conversion_rate.smoothed"]]
            #         conversion_rates = conversion_rates.rename(columns={"conversion_rate.smoothed": "conversion_rate"})
            #
            #     conversion_rate_module = ConversionRateFlatDate(conversion_rates, seed=seed)
            #
            # return conversion_rate_module

            conversion_rate_module = ConversionRateFlatDateHoW(df, seed=seed)

        return conversion_rate_module

    def load_conversions_module_with_prior(self, data, modules_request, seed):
        """
        Loads a date-how conversions module_loader as defined in modules_request
        based on the prior provided in data.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param int seed: Seed for the random number generator.
        :return: action_set conversions module_loader as defined in modules_request.
        :rtype: ConversionsDateHow
        """

        assert "conversion_rate" in data.columns, "There is no conversion_rate column in [data] DataFrame"

        conversions_module = None

        if "ConversionsBinomial" in modules_request["conversions"]["class"]:
            df = self.prepare_conversions_prior_data(data=data, modules_request=modules_request)

            conversions_module = ConversionsBinomialDateHoW(prior=df, seed=seed)

        return conversions_module

    def prepare_conversion_rate_prior_data(self, data, modules_request):
        """
        Prepares a prior for the conversion_rate module.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :return: Prior for the conversions module_loader.
        :rtype: pd.DataFrame
        """

        conversion_rate_df = None

        state_list = ["date", "hour_of_week"]

        if modules_request["conversion_rate"]["conversion_rate"] == "original":
            column_list = state_list.copy()
            column_list.extend(["conversion_rate"])

            conversion_rate_df = data.loc[:, column_list]

        elif modules_request["conversion_rate"]["conversion_rate"] == "smoothed":
            assert "conversion_rate.smoothed" in data.columns, \
                "There is no conversion_rate column in [data] DataFrame"

            column_list = state_list.copy()
            column_list.extend(["conversion_rate.smoothed"])

            conversion_rate_df = data.loc[:, column_list]
            conversion_rate_df = conversion_rate_df.rename(columns={"conversion_rate.smoothed": "conversion_rate"})

        elif modules_request["conversion_rate"]["conversion_rate"] == "decomposed":
            column_list = state_list.copy()
            column_list.extend(["conversion_rate"])

            conversion_rate_df = data.loc[:, column_list]
            conversion_rate_df["conversion_rate"] = self.concatenate_decomposed_series(
                                                                    data=data,
                                                                    params=modules_request["conversion_rate"]["params"])

        return conversion_rate_df

    def prepare_conversions_prior_data(self, data, modules_request):
        """
        Prepares a prior for the conversions module_loader.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :return: Prior for the conversions module_loader.
        :rtype: pd.DataFrame
        """

        state_list = ["date", "hour_of_week"]

        conversions_df = data.loc[:, state_list]

        return conversions_df

    def load_revenue_module_with_prior(self, data, modules_request, seed):
        """
        Loads a date-how revenue module_loader as defined in modules_request
        based on the prior provided in data.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param int seed: Seed for the random number generator.
        :return: action_set revenue module_loader as defined in modules_request.
        :rtype: RevenueDateHow
        """

        assert "value_per_conversion" in data.columns, \
            "There is no value_per_conversion column in [data] DataFrame"

        revenue_module = None

        if "RevenueConversionBased" in modules_request["revenue"]["class"]:
            df = self.prepare_revenue_prior_data(data=data, modules_request=modules_request)

            revenue_module = RevenueConversionBasedDateHoW(avg_rpv=df, seed=seed)

        return revenue_module

    def prepare_revenue_prior_data(self, data, modules_request):
        """
        Prepares a prior for the revenue module_loader.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :return: Prior for the revenue module_loader.
        :rtype: pd.DataFrame
        """

        revenue_df = None

        state_list = ["date", "hour_of_week"]

        if modules_request["revenue"]["value_per_conversion"] == "original":
            column_list = state_list.copy()
            column_list.extend(["value_per_conversion"])

            revenue_df = data.loc[:, column_list]

        elif modules_request["revenue"]["value_per_conversion"] == "smoothed":
            assert "value_per_conversion.smoothed" in data.columns, \
                "There is no value_per_conversion.smoothed column in [data] DataFrame"

            column_list = state_list.copy()
            column_list.extend(["value_per_conversion.smoothed"])

            revenue_df = data.loc[:, column_list]
            revenue_df = revenue_df.rename(columns={"value_per_conversion.smoothed": "value_per_conversion"})

        elif modules_request["revenue"]["value_per_conversion"] == "decomposed":
            column_list = state_list.copy()
            column_list.extend(["value_per_conversion"])

            revenue_df = data.loc[:, column_list]
            revenue_df["value_per_conversion"] = self.concatenate_decomposed_series(
                                                                    data=data,
                                                                    params=modules_request["revenue"]["params"])

        return revenue_df

    def load_cpc_module_with_prior(self, data, modules_request, seed):
        """
        Loads a date-how cost per click module_loader as defined in modules_request
        based on the prior provided in data.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param int seed: Seed for the random number generator.
        :return: action_set cost per click module_loader as defined in modules_request.
        :rtype: CPCDateHowModule
        """

        cpc_module = None

        state_list = ["date", "hour_of_week"]

        if "CPCFirstPrice" in modules_request["cpc"]["class"]:

            cpc_class_module = CPCFirstPriceDateHoW

            cpc_module = cpc_class_module(dates=data.loc[:, state_list])

        elif "CPCSimpleSecondPrice" in modules_request["cpc"]["class"]:

            cpc_class_module = CPCSimpleSecondPriceDateHoW

            cpc_module = cpc_class_module(dates=data.loc[:, state_list], seed=seed)

        elif "CPCBidHistoricalAvgCPC" in modules_request["cpc"]["class"]:
            assert "average_cpc" in data.columns, "There is no average_cpc column in [data] DataFrame"

            average_cpc_df = self.prepare_average_cpc_prior_data(data=data,
                                                                 modules_request=modules_request)

            cpc_module = CPCBidHistoricalAvgCPCDateHoW(mu_cpc=average_cpc_df, seed=seed)

        elif modules_request["cpc"]["class"] == "CPCBidMinusCpcDiffDateHoW":
            assert "weighted_bid" in data.columns, "There is no weighted_bid column in [data] DataFrame"
            assert "average_cpc" in data.columns, "There is no average_cpc column in [data] DataFrame"

            bid_df = self.prepare_weighted_bid_prior_data(data=data, modules_request=modules_request,
                                                          variable_name="cpc")

            average_cpc_df = self.prepare_average_cpc_prior_data(data=data,
                                                                 modules_request=modules_request)

            cpc_module = CPCBidMinusCpcDiffDateHoW(avg_hist_bid=bid_df, avg_hist_cpc=average_cpc_df, seed=seed)

        return cpc_module

    def prepare_average_cpc_prior_data(self, data, modules_request):
        """
        Prepares a prior for the cost per click module_loader.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :return: Prior for the cost per click module_loader.
        :rtype: pd.DataFrame
        """

        average_cpc_df = None

        state_list = ["date", "hour_of_week"]

        if modules_request["cpc"]["average_cpc"] == "original":
            column_list = state_list.copy()
            column_list.extend(["average_cpc"])

            average_cpc_df = data.loc[:, column_list]

        elif modules_request["cpc"]["average_cpc"] == "smoothed":
            assert "average_cpc.smoothed" in data.columns, \
                "There is no average_cpc.smoothed column in [data] DataFrame"

            column_list = state_list.copy()
            column_list.extend(["average_cpc.smoothed"])

            average_cpc_df = data.loc[:, column_list]
            average_cpc_df = average_cpc_df.rename(columns={"average_cpc.smoothed": "average_cpc"})

        elif modules_request["cpc"]["average_cpc"] == "decomposed":
            column_list = state_list.copy()
            column_list.extend(["average_cpc"])

            average_cpc_df = data.loc[:, column_list]
            average_cpc_df["average_cpc"] = self.concatenate_decomposed_series(
                                                        data=data,
                                                        params=modules_request["cpc"]["params"]["average_cpc_params"])

        return average_cpc_df
