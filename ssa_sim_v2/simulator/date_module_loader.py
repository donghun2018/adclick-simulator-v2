# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------
    
import numpy as np
import pandas as pd

from ssa_sim_v2.simulator.module_loader import ModuleLoader

from ssa_sim_v2.simulator.modules.auctions.auctions_date import AuctionsPoissonDate

from ssa_sim_v2.simulator.modules.clicks.clicks_date import ClicksBinomialClickProbModelDate

from ssa_sim_v2.simulator.modules.click_probability.click_probability_date import ClickProbabilityModelDate
from ssa_sim_v2.simulator.modules.click_probability.click_probability_date import ClickProbabilityLogisticDate
from ssa_sim_v2.simulator.modules.click_probability.click_probability_date import ClickProbabilityLogisticLogDate

from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_date import ConversionRateFlatDate

from ssa_sim_v2.simulator.modules.conversions.conversions_date import ConversionsBinomialDate

from ssa_sim_v2.simulator.modules.revenue.revenue_date import RevenueConversionBasedDate

from ssa_sim_v2.simulator.modules.rpc.rpc_date import RPCHistoricalAvgDate

from ssa_sim_v2.simulator.modules.cpc.cpc_date import CPCFirstPriceDate
from ssa_sim_v2.simulator.modules.cpc.cpc_date import CPCSimpleSecondPriceDate
from ssa_sim_v2.simulator.modules.cpc.cpc_date import CPCBidHistoricalAvgCPCDate
from ssa_sim_v2.simulator.modules.cpc.cpc_date import CPCBidMinusCpcDiffDate

from ssa_sim_v2.simulator.modules.avg_price.avg_price_date import AvgPriceHistoricalAvgDate

from ssa_sim_v2.simulator.modules.average_position.average_position_date import AveragePositionHyperbolicDate

from ssa_sim_v2.prior_generators.auctions.date import *
from ssa_sim_v2.prior_generators.avg_price.date import *
from ssa_sim_v2.prior_generators.click_probability.date import *
from ssa_sim_v2.prior_generators.revenue.date import *
from ssa_sim_v2.prior_generators.cpc.date import *
from ssa_sim_v2.prior_generators.conversion_rate.date import *
from ssa_sim_v2.prior_generators.conversions.date import *

# ------------------------------------------------------------


class DateModuleLoader(ModuleLoader):
    """
    Module loader providing load_modules method which returns a dictionary
    with date modules initialized from provided real priors or prior generators
    according to an experiment spec.
    """

    def __init__(self):
        ModuleLoader.__init__(self)
    
    def load_modules(self, experiment_spec, data, seed=12345):
        """
        Loads date modules according to the experiment spec.

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

        # Prepare modules
        self.modules = {}
        
        if "auctions" in modules_request.keys():

            self.modules["auctions"] = self.load_auctions_module(data, modules_request,
                                                                 date_from, date_to, seed)
            
        if "avg_price" in modules_request.keys():            

            self.modules["avg_price"] \
                = self.load_avg_price_module(data, modules_request,
                                             date_from, date_to, seed)
            
        if "clicks" in modules_request.keys():

            self.modules["clicks"], self.modules["click_probability"] \
                = self.load_clicks_module(data, modules_request,
                                          date_from, date_to, seed)

        if "conversion_rate" in modules_request.keys():

            self.modules["conversion_rate"] \
                = self.load_conversion_rate_module(data, modules_request,
                                                   date_from, date_to, seed)

        if "conversions" in modules_request.keys():

            self.modules["conversions"] \
                = self.load_conversions_module(data, modules_request,
                                               date_from, date_to, seed)
            
        if "cpc" in modules_request.keys():

            self.modules["cpc"] = self.load_cpc_module(data, modules_request,
                                                       date_from, date_to, seed)
            
        if "revenue" in modules_request.keys():

            self.modules["revenue"] = self.load_revenue_module(data, modules_request,
                                                               date_from, date_to, seed)

        if "average_position" in modules_request.keys():  

            self.modules["average_position"] \
                = self.load_average_position_module(data, modules_request,
                                                    date_from, date_to, seed)

        return self.modules

    def load_module_using_prior_generator(self, variable_name, modules_request,
                                          date_from, date_to, seed, class_key="class"):
        """
        Loads an appropriate module for the variable variable_name using a prior
        from the synthetic module generator defined in modules_request.

        :param str variable_name: The variable name for which a module_loader is to be
            created. simulator.g. avg_price, cpc, conversions etc.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.
        :param str class_key:
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

    def load_auctions_module(self, data, modules_request, date_from, date_to, seed):
        """
        Loads a date auctions module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.
        :return: An auctions module as defined in modules_request.
        :rtype: AuctionsDate
        """

        if "Generator" in modules_request["auctions"]["class"]:

            auctions_module = self.load_module_using_prior_generator("auctions",
                                                                     modules_request,
                                                                     date_from, date_to, seed)

        elif modules_request["auctions"]["class"] == "AuctionsPoissonDate":

            assert data is not None, self.data_input_assert_message
            assert("auctions" in data.columns)

            if modules_request["auctions"]["auctions"] == "original":
                auctions = data.loc[:, ["date", "auctions"]]
            elif modules_request["auctions"]["auctions"] == "smoothed":
                assert("auctions.smoothed" in data.columns)
                auctions = data.loc[:, ["date", "auctions.smoothed"]]
                auctions = auctions.rename(columns={"auctions.smoothed": "auctions"})

            auctions_module = AuctionsPoissonDate(auctions, seed=seed)

        return auctions_module

    def load_avg_price_module(self, data, modules_request, date_from, date_to, seed):
        """
        Loads a date average price module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.
        :return: action_set date average price module as defined in modules_request.
        :rtype: AvgPriceDate
        """

        if "Generator" in modules_request["avg_price"]["class"]:

            avg_price_module = self.load_module_using_prior_generator("avg_price",
                                                                      modules_request,
                                                                      date_from, date_to, seed)

        elif modules_request["avg_price"]["class"] == "AvgPriceHistoricalAvgDate":

            assert data is not None, self.data_input_assert_message
            assert("avg_price" in data.columns)

            if modules_request["avg_price"]["avg_price"] == "original":
                avg_price = data.loc[:, ["date", "avg_price"]]

            if modules_request["avg_price"]["avg_price"] == "smoothed":
                assert("avg_price.smoothed" in data.columns)
                avg_price = data.loc[:, ["date", "avg_price.smoothed"]]
                avg_price = avg_price.rename(columns={"avg_price.smoothed": "avg_price"})

            avg_price_module = AvgPriceHistoricalAvgDate(avg_price, seed=seed)

        return avg_price_module

    def load_clicks_module(self, data, modules_request, date_from, date_to, seed):
        """
        Loads a date clicks module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.
        :return: action_set date clicks module as defined in modules_request.
        :rtype: ClicksDate
        """

        if modules_request["clicks"]["class"] == "ClicksBinomialClickProbModelDate":

            click_prob_module = None

            if "Generator" in modules_request["clicks"]["cp_class"]:

                click_prob_module = self.load_module_using_prior_generator("clicks",
                                                                           modules_request,
                                                                           date_from, date_to, seed,
                                                                           class_key="cp_class")

            else:

                assert data is not None, self.data_input_assert_message
                assert("click_probability" in data.columns)
                assert("weighted_bid" in data.columns)

                if modules_request["clicks"]["cp_class"] == "ClickProbabilityLogisticDate":
                    click_prob_class = ClickProbabilityLogisticDate

                elif modules_request["clicks"]["cp_class"] == "ClickProbabilityLogisticLogDate":
                    click_prob_class = ClickProbabilityLogisticLogDate

                if modules_request["clicks"]["click_probability"] == "original":
                    click_probability = data.loc[:, ["date", "click_probability"]]
                elif modules_request["clicks"]["click_probability"] == "smoothed":
                    assert("click_probability.smoothed" in data.columns)
                    click_probability = data.loc[:, ["date", "click_probability.smoothed"]]
                    click_probability = click_probability.rename(columns={"click_probability.smoothed":"click_probability"})

                if modules_request["clicks"]["weighted_bid"] == "original":
                    bid = data.loc[:, ["date", "weighted_bid"]]
                    bid = bid.rename(columns={"weighted_bid":"cpc_bid"})
                elif modules_request["clicks"]["weighted_bid"] == "smoothed":
                    assert("weighted_bid.smoothed" in data.columns)
                    bid = data.loc[:, ["date", "weighted_bid.smoothed"]]
                    bid = bid.rename(columns={"weighted_bid.smoothed": "cpc_bid"})

                click_prob_module = click_prob_class(click_probability, bid, seed=seed)

            clicks_module = ClicksBinomialClickProbModelDate(click_prob_module, seed=seed)

        return clicks_module, click_prob_module

    def load_conversion_rate_module(self, data, modules_request, date_from, date_to, seed):
        """
        Loads a conversion rate module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.
        :return: action_set conversion rate module as defined in modules_request.
        :rtype: ConversionRateDate
        """

        if "Generator" in modules_request["conversion_rate"]["class"]:

            conversion_rate_module = self.load_module_using_prior_generator("conversion_rate",
                                                                            modules_request,
                                                                            date_from, date_to, seed)

        elif modules_request["conversion_rate"]["class"] == "ConversionRateFlatDate":

            assert data is not None, self.data_input_assert_message
            assert("conversion_rate" in data.columns)

            if modules_request["conversion_rate"]["conversion_rate"] == "original":
                conversion_rates = data.loc[:, ["date", "conversion_rate"]]
            elif modules_request["conversion_rate"]["conversion_rate"] == "smoothed":
                assert("conversion_rate.smoothed" in data.columns)
                conversion_rates = data.loc[:, ["date", "conversion_rate.smoothed"]]
                conversion_rates = conversion_rates.rename(columns={"conversion_rate.smoothed": "conversion_rate"})

            conversion_rate_module = ConversionRateFlatDate(conversion_rates, seed=seed)

        return conversion_rate_module

    def load_conversions_module(self, data, modules_request, date_from, date_to, seed):
        """
        Loads a conversions module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.
        :return: action_set conversions module as defined in modules_request.
        :rtype: ConversionsDate
        """

        if "Generator" in modules_request["conversions"]["class"]:

            conversions_module = self.load_module_using_prior_generator("conversions",
                                                                        modules_request,
                                                                        date_from, date_to, seed)

        elif modules_request["conversions"]["class"] == "ConversionsBinomialDate":

            assert data is not None, self.data_input_assert_message

            dates = data.loc[:, ["date"]]

            conversions_module = ConversionsBinomialDate(dates, seed=seed)

        return conversions_module

    def load_cpc_module(self, data, modules_request, date_from, date_to, seed):
        """
        Loads a cpc module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.
        :return: action_set cpc module as defined in modules_request.
        :rtype: CpcDate
        """

        if "Generator" in modules_request["cpc"]["class"]:

            cpc_module = self.load_module_using_prior_generator("cpc",
                                                                modules_request,
                                                                date_from, date_to, seed)

        elif modules_request["cpc"]["class"] == "CPCFirstPriceDate":

            assert data is not None, self.data_input_assert_message
            cpc_module = CPCFirstPriceDate(data.loc[:, ["date"]])

        elif modules_request["cpc"]["class"] == "CPCSimpleSecondPriceDate":

            assert data is not None, self.data_input_assert_message
            cpc_module = CPCSimpleSecondPriceDate(data.loc[:, ["date"]], seed=seed)

        elif modules_request["cpc"]["class"] == "CPCBidHistoricalAvgCPCDate":

            assert data is not None, self.data_input_assert_message

            if modules_request["cpc"]["average_cpc"] == "original":
                assert("average_cpc" in data.columns)
                cpc = data.loc[:, ["date", "average_cpc"]]

            elif modules_request["cpc"]["average_cpc"] == "smoothed":
                assert("average_cpc.smoothed" in data.columns)
                cpc = data.loc[:, ["date", "average_cpc.smoothed"]]
                cpc = cpc.rename(columns={"average_cpc.smoothed": "average_cpc"})

            cpc_module = CPCBidHistoricalAvgCPCDate(cpc, seed=seed)

        elif modules_request["cpc"]["class"] == "CPCBidMinusCpcDiffDate":

            assert data is not None, self.data_input_assert_message

            if modules_request["cpc"]["weighted_bid"] == "original":
                assert("weighted_bid" in data.columns)
                bid = data.loc[:, ["date", "weighted_bid"]]
                bid = bid.rename(columns={"weighted_bid":"cpc_bid"})
            elif modules_request["cpc"]["weighted_bid"] == "smoothed":
                assert("weighted_bid.smoothed" in data.columns)
                bid = data.loc[:, ["date", "weighted_bid.smoothed"]]
                bid = bid.rename(columns={"weighted_bid.smoothed": "cpc_bid"})

            if modules_request["cpc"]["average_cpc"] == "original":
                assert("average_cpc" in data.columns)
                average_cpc = data.loc[:, ["date", "average_cpc"]]
            elif modules_request["cpc"]["average_cpc"] == "smoothed":
                assert("average_cpc.smoothed" in data.columns)
                average_cpc = data.loc[:, ["date", "average_cpc.smoothed"]]
                average_cpc = average_cpc.rename(columns={"average_cpc.smoothed": "average_cpc"})

            cpc_module = CPCBidMinusCpcDiffDate(bid, average_cpc, seed=seed)

        return cpc_module

    def load_revenue_module(self, data, modules_request, date_from, date_to, seed):
        """
        Loads a revenue module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.
        :return: action_set revenue module as defined in modules_request.
        :rtype: RevenueDate
        """

        if "Generator" in modules_request["revenue"]["class"]:

            revenue_module = self.load_module_using_prior_generator("revenue",
                                                                    modules_request,
                                                                    date_from, date_to, seed)

        elif modules_request["revenue"]["class"] == "RevenueConversionBasedDate":

            assert data is not None, self.data_input_assert_message

            if modules_request["revenue"]["value_per_conversion"] == "original":
                assert("value_per_conversion" in data.columns)
                value_per_conversion = data.loc[:, ["date", "value_per_conversion"]]

            elif modules_request["revenue"]["value_per_conversion"] == "smoothed":
                assert("value_per_conversion.smoothed" in data.columns)
                value_per_conversion = data.loc[:, ["date", "value_per_conversion.smoothed"]]
                value_per_conversion = value_per_conversion.rename(columns={"value_per_conversion.smoothed":"value_per_conversion"})

            revenue_module = RevenueConversionBasedDate(value_per_conversion, seed=seed)

        return revenue_module

    def load_average_position_module(self, data, modules_request, date_from, date_to, seed):
        """
        Loads a date average position module as defined in modules_request.

        :param pd.DataFrame data: Input data.
        :param dict modules_request: action_set dictionary with a modules specification.
        :param str date_from: Starting date.
        :param str date_to: End date.
        :param int seed: Seed for the random number generator.
        :return: action_set date average position module as defined in modules_request.
        :rtype: AveragePositionDate
        """

        if modules_request["average_position"]["class"] == "AveragePositionHyperbolicDate":

            assert data is not None, self.data_input_assert_message
            assert("click_probability" in self.modules.keys())

            max_cp = data.loc[(data["date"] >= date_from) & (data["date"] <= date_to), ["date"]].reset_index()
            max_cp["max_cp"] = pd.Series([self.modules["click_probability"].get_cp(float("inf"), max_cp["date"][t])
                                          for t in range(len(max_cp))])

            average_position_module = AveragePositionHyperbolicDate(max_cp, seed=seed)

        return average_position_module
