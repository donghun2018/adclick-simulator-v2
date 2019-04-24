# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from ssa_sim_v2.tools.file_data_handler import FileDataHandler
from ssa_sim_v2.simulator.how_module_loader import OneWeekHoWModuleLoader
from ssa_sim_v2.simulator.date_module_loader import DateModuleLoader
from ssa_sim_v2.simulator.date_how_module_loader import DateHoWModuleLoader

# ------------------------------------------------------------


class CSVModuleLoader(object):

    def __init__(self):
        pass

    def load_one_week_how_modules(self, experiment_spec, seed=12345):
        
        dataset_name = experiment_spec["dataset_name"]
        
        file_data_handler = FileDataHandler()

        data = file_data_handler.load_data("", dataset_name)
        
        module_loader = OneWeekHoWModuleLoader()

        return module_loader.load_modules(experiment_spec, data, seed)
    
    def load_date_modules(self, experiment_spec, seed=12345):
        
        dataset_name = experiment_spec["dataset_name"]

        file_data_handler = FileDataHandler()

        data = file_data_handler.load_data("", dataset_name)

        module_loader = DateModuleLoader()

        return module_loader.load_modules(experiment_spec, data, seed)

    def load_date_how_modules(self, experiment_spec, seed=12345):

        dataset_name = experiment_spec["dataset_name"]

        file_data_handler = FileDataHandler()

        data = file_data_handler.load_data("", dataset_name)

        module_loader = DateHoWModuleLoader()

        return module_loader.load_modules(experiment_spec, data, seed)
