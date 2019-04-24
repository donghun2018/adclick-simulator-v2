# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------
    
import numpy as np
import pandas as pd

from ssa_sim_v2.simulator.module_loader import ModuleLoader

from ssa_sim_v2.simulator.modules.auctions.auctions_date_how import AuctionsPoissonDateHoW

from ssa_sim_v2.simulator.modules.clicks.clicks_date_how import ClicksBinomialClickProbModelDateHoW

from ssa_sim_v2.simulator.modules.click_probability.click_probability_date_how import ClickProbabilityModelDateHoW
from ssa_sim_v2.simulator.modules.click_probability.click_probability_date_how import ClickProbabilityLogisticDateHoW
from ssa_sim_v2.simulator.modules.click_probability.click_probability_date_how import ClickProbabilityLogisticLogDateHoW

from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_date_how \
    import (ConversionRateDateHoW, ConversionRateFlatDateHoW, ConversionRateFunctionDateHoW)

from ssa_sim_v2.simulator.modules.conversions.conversions_date_how import ConversionsBinomialDateHoW

from ssa_sim_v2.simulator.modules.revenue.revenue_date_how import RevenueConversionBasedDateHoW

from ssa_sim_v2.simulator.modules.rpc.rpc_date_how import RPCHistoricalAvgDateHoW

from ssa_sim_v2.simulator.modules.cpc.cpc_date_how import CPCFirstPriceDateHoW
from ssa_sim_v2.simulator.modules.cpc.cpc_date_how import CPCSimpleSecondPriceDateHoW
from ssa_sim_v2.simulator.modules.cpc.cpc_date_how import CPCBidHistoricalAvgCPCDateHoW
from ssa_sim_v2.simulator.modules.cpc.cpc_date_how import CPCBidMinusCpcDiffDateHoW

from ssa_sim_v2.simulator.modules.avg_price.avg_price_date_how import AvgPriceHistoricalAvgDateHoW

from ssa_sim_v2.simulator.modules.average_position.average_position_date_how import AveragePositionHyperbolicDateHoW

from ssa_sim_v2.prior_generators.auctions.date_how import *
from ssa_sim_v2.prior_generators.avg_price.date_how import *
from ssa_sim_v2.prior_generators.click_probability.date_how import *
from ssa_sim_v2.prior_generators.revenue.date_how import *
from ssa_sim_v2.prior_generators.cpc.date_how import *
from ssa_sim_v2.prior_generators.conversion_rate.date_how import *
from ssa_sim_v2.prior_generators.conversions.date_how import *

# ------------------------------------------------------------


class DateHoWModuleLoader(ModuleLoader):
    """
    Module loader providing load_modules method which returns a dictionary
    with date-hour of week modules initialized from provided real priors
    or prior generators according to an experiment spec.
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
        # Add "hour_of_week" column if not exist in data
        if (type(data) == pd.DataFrame) and ("hour_of_week" not in data.columns) and ("hour_of_day" in data.columns):
            data["hour_of_week"] = pd.to_datetime(data["date"]).dt.dayofweek * 24 + data["hour_of_day"]

        modules_request_keys = modules_request.keys()

        if "auctions" in modules_request_keys:

            self.modules["auctions"] = self.load_auctions_module(data=data, modules_request=modules_request,
                                                                 date_from=date_from, date_to=date_to, seed=seed)

        if "avg_price" in modules_request_keys:

            self.modules["avg_price"] = self.load_avg_price_module(data=data, modules_request=modules_request,
                                                                   date_from=date_from, date_to=date_to, seed=seed)

        if "clicks" in modules_request_keys:
            self.modules["click_probability"], self.modules["clicks"] = self.load_click_probability_and_clicks_module(
                                                                        data=data, modules_request=modules_request,
                                                                        date_from=date_from, date_to=date_to, seed=seed)

        if "conversion_rate" in modules_request_keys:
            self.modules["conversion_rate"] = self.load_conversion_rate_module(data=data,
                                                                               modules_request=modules_request,
                                                                               date_from=date_from, date_to=date_to,
                                                                               seed=seed)

        if "conversions" in modules_request_keys:
            self.modules["conversions"] = self.load_conversions_module(data=data, modules_request=modules_request,
                                                                       date_from=date_from, date_to=date_to, seed=seed)

        if "cpc" in modules_request_keys:
            self.modules["cpc"] = self.load_cpc_module(data=data, modules_request=modules_request, date_from=date_from,
                                                       date_to=date_to, seed=seed)

        if "revenue" in modules_request_keys:
            self.modules["revenue"] = self.load_revenue_module(data=data, modules_request=modules_request,
                                                               date_from=date_from, date_to=date_to, seed=seed)

        if "average_position" in modules_request_keys:
            self.modules["average_position"] = self.load_average_position_module(data=data,
                                                                                 modules_request=modules_request,
                                                                                 date_from=date_from, date_to=date_to,
                                                                                 seed=seed)

        return self.modules

    def load_auctions_module(self, data, modules_request, date_from, date_to, seed):
        """
        Load a date with hour of week auctions module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: A dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.

        :return: A date with hour of week auctions module_loader as defined in modules_request.
        :rtype: AuctionsDateHoWModule
        """

        series_name = "auctions"

        # Settings for artificial generators
        if "Generator" in modules_request[series_name]["class"]:
            auctions_module = self.load_module_using_prior_generator(variable_name=series_name,
                                                                     modules_request=modules_request,
                                                                     date_from=date_from,
                                                                     date_to=date_to,
                                                                     seed=seed)
        # Settings for statistical modules
        else:
            auctions_module = self.load_auctions_module_with_prior(data=data,
                                                                   modules_request=modules_request,
                                                                   seed=seed)

        return auctions_module

    def load_avg_price_module(self, data, modules_request, date_from, date_to, seed):
        """
        Load a date with hour of week avg price module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: A dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.

        :return: A date with hour of week avg price module_loader as defined in modules_request.
        :rtype: AvgPriceDateHoW
        """

        series_name = "avg_price"

        # Settings for artificial generators
        if "Generator" in modules_request[series_name]["class"]:
            avg_price_module = self.load_module_using_prior_generator(variable_name=series_name,
                                                                      modules_request=modules_request,
                                                                      date_from=date_from,
                                                                      date_to=date_to,
                                                                      seed=seed)
        # Settings for statistical modules
        else:
            avg_price_module = self.load_avg_price_module_with_prior(data=data,
                                                                     modules_request=modules_request,
                                                                     seed=seed)

        return avg_price_module

    def load_click_probability_and_clicks_module(self, data, modules_request, date_from, date_to, seed):
        """
        Loads a date with hour of week click probabilities and clicks module_loader as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: A dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.

        :return: Two date with hour of week click probability and clicks module_loaders as defined in modules_request.
        :rtype: ClickProbabilityDateHoWModule, ClicksDateHoWModule
        """
        series_name = "clicks"
        click_prob_module = None
        clicks_module = None

        if modules_request["clicks"]["class"] == "ClicksBinomialClickProbModelDateHoW":

            # Settings for artificial generators
            if "Generator" in modules_request[series_name]["cp_class"]:
                click_prob_module = self.load_module_using_prior_generator(variable_name=series_name,
                                                                           modules_request=modules_request,
                                                                           date_from=date_from,
                                                                           date_to=date_to,
                                                                           seed=seed,
                                                                           class_key="cp_class")
            # Settings for statistical modules
            elif "theta_1" in modules_request[series_name].keys():
                pass
            else:
                click_prob_module = self.load_click_probability_module_with_prior(
                    data=data, modules_request=modules_request, seed=seed)

            clicks_module = ClicksBinomialClickProbModelDateHoW(p=click_prob_module, seed=seed)

        return click_prob_module, clicks_module

    def load_conversion_rate_module(self, data, modules_request, date_from, date_to, seed):
        """
        Load a conversion rate module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: A dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.

        :return: action_set conversion_rate module as defined in modules_request.
        :rtype: ConversionRateDateHoWModule
        """

        if "Generator" in modules_request["conversion_rate"]["class"]:
            conversion_rate_module = self.load_module_using_prior_generator("conversion_rate",
                                                                            modules_request,
                                                                            date_from, date_to, seed)
        else:
            conversion_rate_module = self.load_conversion_rate_module_with_prior(data=data,
                                                                                 modules_request=modules_request,
                                                                                 seed=seed)

        return conversion_rate_module

    def load_conversions_module(self, data, modules_request, date_from, date_to, seed):
        """
        Loads a date-how conversions module as defined in modules_request
        based on the prior provided in data.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: A dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.

        :return: action_set conversions module as defined in modules_request.
        :rtype: ConversionsDateHow
        """
        series_name = "conversions"

        # Settings for artificial generators
        if "Generator" in modules_request[series_name]["class"]:
            conversions_module = self.load_module_using_prior_generator(variable_name=series_name,
                                                                        modules_request=modules_request,
                                                                        date_from=date_from,
                                                                        date_to=date_to,
                                                                        seed=seed)
        # Settings for statistical modules
        else:
            conversions_module = self.load_conversions_module_with_prior(data=data,
                                                                         modules_request=modules_request,
                                                                         seed=seed)

        return conversions_module

    def load_cpc_module(self, data, modules_request, date_from, date_to, seed):
        series_name = "cpc"

        if "Generator" in modules_request[series_name]["class"]:
            cpc_module = self.load_module_using_prior_generator(variable_name=series_name,
                                                                modules_request=modules_request,
                                                                date_from=date_from,
                                                                date_to=date_to,
                                                                seed=seed)
        # Settings for statistical modules
        else:
            cpc_module = self.load_cpc_module_with_prior(data=data,
                                                         modules_request=modules_request,
                                                         seed=seed)

        return cpc_module

    def load_revenue_module(self, data, modules_request, date_from, date_to, seed):
        series_name = "revenue"

        # Settings for artificial generators
        if "Generator" in modules_request[series_name]["class"]:
            revenue_module = self.load_module_using_prior_generator(variable_name=series_name,
                                                                    modules_request=modules_request,
                                                                    date_from=date_from,
                                                                    date_to=date_to,
                                                                    seed=seed)
        # Settings for statistical modules
        else:
            revenue_module = self.load_revenue_module_with_prior(data=data,
                                                                 modules_request=modules_request,
                                                                 seed=seed)

        return revenue_module

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
        :return: An appropriate module_loader for the variable variable_name using a prior
            from the synthetic module_loader generator defined in modules_request.
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
