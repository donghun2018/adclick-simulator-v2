# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------
    
# Load libraries ---------------------------------------------


# ------------------------------------------------------------


class DataSourcesDefinitions(object):
    
    def __init__(self):
        
        self.gadw_hourly = ["auctions", "impressions", "clicks", "conversions", "positions_sum", "cost", 
                            "total_conversion_value", "search_impression_share", "ctr", "click_probability", 
                            "conversion_rate", "average_position", "average_cpc", "cost_per_conversion", 
                            "value_per_conversion", "value_per_click"]
        self.gadw_params_hourly = ["campaign_status", "ad_group_status", "cpc_bid", "weighted_bid"]
        
        self.gadw_hourly_numerical = ["auctions", "impressions", "clicks", "conversions", "positions_sum", "cost", 
                            "total_conversion_value", "search_impression_share", "ctr", "click_probability", 
                            "conversion_rate", "average_position", "average_cpc", "cost_per_conversion", 
                            "value_per_conversion", "value_per_click"]
        self.gadw_params_hourly_numerical = ["cpc_bid", "weighted_bid"]
        
        self.previo_reservations = ["reservations", "price", "avg_price", "currency", 
                                    "price_plan_price", "avg_price_plan_price", "price_plan_currency"]
        self.previo_engine_reservations = ["engine_reservations", "engine_price", "engine_avg_price", 
                                           "engine_currency", "engine_price_plan_price", 
                                           "engine_avg_price_plan_price", "engine_price_plan_currency"]
        self.previo_price_plan = ["price_plan_price", "avg_price_plan_price", "price_plan_currency",
                                  "engine_price_plan_price", "engine_avg_price_plan_price", "engine_price_plan_currency"]
        
        self.previo_reservations_numerical = ["reservations", "price", "avg_price", 
                                              "price_plan_price", "avg_price_plan_price"]
        self.previo_engine_reservations_numerical = ["engine_reservations", "engine_price", "engine_avg_price", 
                                                     "engine_price_plan_price", "engine_avg_price_plan_price"]
    
    
    def get_gadw_hourly_columns(self):
        
        return self.gadw_hourly
        
    
    def get_gadw_params_hourly_columns(self):
        
        return self.gadw_params_hourly
    
    
    def get_gadw_hourly_numerical_columns(self):
        
        return self.gadw_hourly_numerical
        
    
    def get_gadw_params_hourly_numerical_columns(self):
        
        return self.gadw_params_hourly_numerical
    
    
    def get_previo_reservations_columns(self):
        
        return self.previo_reservations
    
    
    def get_previo_engine_reservations_columns(self):
        
        return self.previo_engine_reservations
    

    def get_previo_price_plan_columns(self):
        
        return self.previo_price_plan

    
    def get_previo_reservations_numerical_columns(self):
        
        return self.previo_reservations_numerical
    
    
    def get_previo_engine_reservations_numerical_columns(self):
        
        return self.previo_engine_reservations_numerical
    
    
    