# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------
    
import numpy as np

from ssa_sim_v2.simulator.modules.multi_state_simulator_module import MultiStateSimulatorModule

# ------------------------------------------------------------


class ModuleLoader(object):
    """
    Base class for module loaders. Every module loader should be inherited from
    this class. A module loader provides load_modules method which returns
    a dictionary with initialized modules (either from a real prior or a synthetic
    prior generator) indexed by variable names.
    """

    def __init__(self):
        self.data_input_assert_message = "Dataset is empty"
        self.dataset_specification_assert_message = "Dataset specification not found"
        self.experiment_spec_assert_message = "Experiment spec is empty"

        self.modules = {}  # type: MultiStateSimulatorModule

    @staticmethod
    def concatenate_decomposed_series(data, params):
        series = np.zeros(data.shape[0])

        if "level" in params.keys():
            assert params["level"] in data.columns, "In [data] DataFrame there is no {} column".format(params["level"])

            series += data[params["level"]]

        if "seasonal" in params.keys():
            for seasonal_name in params["seasonal"]:
                assert seasonal_name in data.columns, "In [data] DataFrame there is no {} column".format(seasonal_name)

                series += data[seasonal_name]

        if "trend" in params.keys():
            assert params["trend"] in data.columns, "In [data] DataFrame there is no {} column".format(params["trend"])

            series += data[params["trend"]]

        series = np.maximum(0, series)

        return series
    
    def load_modules(self, experiment_spec, data, seed=12345):
        """
        Loads modules according to the experiment spec. This method should be
        overridden in subclasses.

        :param dict experiment_spec: Experiment specification.
        :param pd.DataFrame data: Input data.
        :param int seed: Seed for the random number generator.
        :return: action_set dictionary with initialized modules to be used by a simulator.
        :rtype: dict
        """

        return {}
