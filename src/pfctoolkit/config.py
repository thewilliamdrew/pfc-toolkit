"""
Configuration objects to facilitate using the Precomputed Connectome.

"""

import os
import json
import importlib.resources as pkg_resources


class Config:
    def __init__(self, pcc):
        from . import configs

        with pkg_resources.path(configs, f"{pcc}.json") as configfile:
            try:
                with open(str(configfile)) as js:
                    self.config = json.load(js)
                if self.check():
                    print(f"Config {self.config['name']} loaded")
                else:
                    raise OSError(f"Config {self.config['name']} unavailable")
            except FileNotFoundError:
                raise FileNotFoundError(f"PCC config file {configfile} does not exist!")

    def check(self):
        checks = ["avgr", "fz", "t", "combo", "std", "norm", "chunk_idx"]
        return all(list(map(os.path.exists, [self.config[key] for key in checks])))

    def get(self, key):
        return self.config[key]

    def get_config(self):
        return self.config
