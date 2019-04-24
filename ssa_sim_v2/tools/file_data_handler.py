# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import os
import csv
import pandas as pd
import ast

# ------------------------------------------------------------


class FileDataHandler(object):
    
    def __init__(self):
        
        self.input_path = os.path.join("data", "input")
        self.output_path = os.path.join("data", "output")
        self.dataset_specs_path = "dataset_specs"
    
    def get_input_path(self):
        return self.input_path 
        
    def get_input_full_path(self, filename):
        return os.path.join(self.get_input_path(), filename)
    
    def get_output_path(self):
        return self.output_path 
    
    def get_output_full_path(self, filename):
        return os.path.join(self.get_output_path(), filename)
    
    def save_data(self, folder, filename, data):
        assert(filename != "")
        if folder != "":
            full_file_path = os.path.join(self.get_input_path(), "{}.csv".format(os.path.join(folder, filename)))
        else:
            full_file_path = os.path.join(self.get_input_path(), "{}.csv".format(filename))
            
        data.to_csv(full_file_path, index=False, quoting=csv.QUOTE_NONE)
    
    def load_data(self, folder, filename):
        assert(filename != "")
        if folder != "":
            full_file_path = os.path.join(self.get_input_path(), "{}.csv".format(os.path.join(folder, filename)))
        else:
            full_file_path = os.path.join(self.get_input_path(), "{}.csv".format(filename))
            
        data = pd.read_csv(full_file_path)
                
        return data
    
    def save_spec(self, folder, filename, spec):
        assert(filename != "")
        if folder != "":
            full_file_path = "{}.spec".format(os.path.join(folder, filename))
        else:
            full_file_path = "{}.spec".format(filename)
        with open(full_file_path, "w") as spec_file:
            spec_file.write(str(spec))
    
    def load_spec(self, folder, filename):
        assert(filename != "")
        if folder != "":
            full_file_path = "{}.spec".format(os.path.join(folder, filename))
        else:
            full_file_path = "{}.spec".format(filename)
        with open(full_file_path, "r") as spec_file:
            spec = ast.literal_eval(spec_file.read())
                
        return spec
    
    def save_dataset_spec(self, folder, filename, spec):
        self.save_spec(os.path.join(self.dataset_specs_path, folder),
                       filename, spec)

    def load_dataset_spec(self, folder, filename):
        return self.load_spec(os.path.join(self.dataset_specs_path, folder),
                              filename)
