# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------
    
import numpy as np
import pandas as pd

from ssa_sim_v2.simulator.module_loader import ModuleLoader

from ssa_sim_v2.tools.data_sources_definitions import DataSourcesDefinitions

from ssa_sim_v2.simulator.modules.auctions.auctions_how import AuctionsPoissonHoW

from ssa_sim_v2.simulator.modules.clicks.clicks_how import ClicksBinomialClickProbModelHoW

from ssa_sim_v2.simulator.modules.click_probability.click_probability_how import ClickProbabilityModelHoW
from ssa_sim_v2.simulator.modules.click_probability.click_probability_how import ClickProbabilityLogisticHoW
from ssa_sim_v2.simulator.modules.click_probability.click_probability_how import ClickProbabilityLogisticLogHoW

from ssa_sim_v2.simulator.modules.conversion_rate.conversion_rate_how import ConversionRateHoW

from ssa_sim_v2.simulator.modules.conversions.conversions_how import ConversionsBinomialHoW

from ssa_sim_v2.simulator.modules.revenue.revenue_how import RevenueConversionBasedHoW

from ssa_sim_v2.simulator.modules.rpc.rpc_how import RPCHistoricalAvgHoW

from ssa_sim_v2.simulator.modules.cpc.cpc_how import CPCFirstPriceHoW
from ssa_sim_v2.simulator.modules.cpc.cpc_how import CPCSimpleSecondPriceHoW
from ssa_sim_v2.simulator.modules.cpc.cpc_how import CPCBidHistoricalAvgCPCHoW
from ssa_sim_v2.simulator.modules.cpc.cpc_how import CPCBidMinusCpcDiffHoW

from ssa_sim_v2.simulator.modules.avg_price.avg_price_how import AvgPriceHistoricalAvgHoW

from ssa_sim_v2.simulator.modules.average_position.average_position_how import AveragePositionHyperbolicHoW

# ------------------------------------------------------------


class OneWeekHoWModuleLoader(ModuleLoader):
    """
    Module loader providing load_modules method which returns a dictionary
    with how modules initialized from provided real priors. These modules
    repeat the same prior every week (168 states).
    """

    def __init__(self):
        ModuleLoader.__init__(self)
    
    def load_modules(self, experiment_spec, data, seed=12345):
        """
        Loads HoW modules according to the experiment spec.

        :param dict experiment_spec: Experiment specification.
        :param pd.DataFrame data: Input data.
        :param int seed: Seed for the random number generator.
        :return: action_set dictionary with initialized modules to be used by a simulator.
        :rtype: dict
        """
        
        modules_request = experiment_spec["modules_request"]
        
        # Choose only seasonal columns appearing in data
        
        data_sources_definitions = DataSourcesDefinitions()
        
        how_columns = ["hour_of_week"]
        
        for column in list(set(data_sources_definitions.get_gadw_hourly_numerical_columns()) \
            | set(data_sources_definitions.get_gadw_params_hourly_numerical_columns()) \
            | set(data_sources_definitions.get_previo_reservations_numerical_columns()) \
            | set(data_sources_definitions.get_previo_engine_reservations_numerical_columns())):
            how_columns.append(column + ".s.how")
        
        columns = list(set(data.columns) & set(how_columns))
        
        print("In module_loader loading")
        
        data = data[columns]
        
        # Get 168 hour of week seasonals
        
        data = data.groupby("hour_of_week").first().reset_index()
        
        print("In module_loader loading - post grouping")
        
        # Prepare modules
        
        self.modules = {}
        auctions_module = None
        clicks_module = None
        conversions_module = None
        click_prob_model = None
        cpc_module = None
        revenue_module = None
        rpc_module = None
        
        if "auctions" in modules_request.keys():
            assert("auctions.s.how" in data.columns)
            
            if modules_request["auctions"] == "AuctionsPoissonHoW":
            
                auctions_module = AuctionsPoissonHoW(data.loc[:, "auctions.s.how"].tolist(), seed=seed)
                
            self.modules["auctions"] = auctions_module
            
        if "clicks" in modules_request.keys():

            if modules_request["clicks"] == "ClicksBinomialClickProbModelHoW":
                
                assert("click_probability.s.how" in data.columns)
                assert("cpc_bid.s.how" in data.columns)
                
                click_prob_model = None
                
                if modules_request["click_probability"] == "ClickProbabilityLogisticHoW":
                    click_prob_model = ClickProbabilityLogisticHoW(data.loc[:, "click_probability.s.how"].tolist(), 
                                                                   data.loc[:, "cpc_bid.s.how"].tolist())
                
                elif modules_request["click_probability"] == "ClickProbabilityLogisticLogHoW":
                    click_prob_model = ClickProbabilityLogisticLogHoW(data.loc[:, "click_probability.s.how"].tolist(), 
                                                                      data.loc[:, "cpc_bid.s.how"].tolist())
            
                clicks_module = ClicksBinomialClickProbModelHoW(click_prob_model, seed=seed)
                
            self.modules["clicks"] = clicks_module
            
        if "conversions" in modules_request.keys():
            assert("conversion_rate.s.how" in data.columns)
            
            if modules_request["conversions"] == "ConversionsBinomialHoW":
            
                conversions_module = ConversionsBinomialHoW(data.loc[:, "conversion_rate.s.how"].tolist(), seed=seed)
                
            self.modules["conversions"] = conversions_module
            
        if "click_probability" in modules_request.keys():
            
            if click_prob_model is None:
                
                assert("click_probability.s.how" in data.columns)
                assert("cpc_bid.s.how" in data.columns)
                
                if modules_request["click_probability"] == "ClickProbabilityLogisticHoW":
                
                    click_prob_model = ClickProbabilityLogisticHoW(data.loc[:, "click_probability.s.how"].tolist(), 
                                                                   data.loc[:, "cpc_bid.s.how"].tolist(), seed=seed)
                    
                elif modules_request["click_probability"] == "ClickProbabilityLogisticLogHoW":
                
                    click_prob_model = ClickProbabilityLogisticLogHoW(data.loc[:, "click_probability.s.how"].tolist(), 
                                                                      data.loc[:, "cpc_bid.s.how"].tolist(), seed=seed)
                    
                self.modules["click_probability"] = click_prob_model
                
            else:
                self.modules["click_probability"] = click_prob_model
            
        if "cpc" in modules_request.keys():
            
            if modules_request["cpc"] == "CPCFirstPriceHoW":
                
                cpc_module = CPCFirstPriceHoW()
                
            elif modules_request["cpc"] == "CPCSimpleSecondPriceHoW":
            
                cpc_module = CPCSimpleSecondPriceHoW(seed=seed)
            
            elif modules_request["cpc"] == "CPCBidHistoricalAvgCPCHoW":
                
                assert("average_cpc.s.how" in data.columns)
            
                cpc_module = CPCBidHistoricalAvgCPCHoW(data.loc[:, "average_cpc.s.how"].tolist(), seed=seed)
                
            elif modules_request["cpc"] == "CPCBidMinusCpcDiffHoW":
                
                assert("cpc_bid.s.how" in data.columns)
                assert("average_cpc.s.how" in data.columns)
            
                cpc_module = CPCBidMinusCpcDiffHoW(data.loc[:, "cpc_bid.s.how"].tolist(),
                                                   data.loc[:, "average_cpc.s.how"].tolist(), seed=seed)
                
            self.modules["cpc"] = cpc_module
            
        if "revenue" in modules_request.keys():            
            assert("value_per_conversion.s.how" in data.columns)
            
            if modules_request["revenue"] == "RevenueConversionBasedHoW":
            
                revenue_module = RevenueConversionBasedHoW(data.loc[:, "value_per_conversion.s.how"].tolist(),
                                                           seed=seed)
                
            self.modules["revenue"] = revenue_module
            
        if "rpc" in modules_request.keys():            
            assert("value_per_click.s.how" in data.columns)
            
            if modules_request["rpc"] == "RPCHistoricalAvgHoW":
            
                rpc_module = RPCHistoricalAvgHoW(data.loc[:, "value_per_click.s.how"].tolist(), seed=seed)
                
            self.modules["rpc"] = rpc_module

        return self.modules
