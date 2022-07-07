"""
Configuration objects to facilitate using the Precomputed Connectome.

"""

import os
import json


class Config:
    def __init__(self, pcc):
        """Generate Config object.

        Parameters
        ----------
        pcc : str
            Name of configfile. Must exist in $HOME/pfctoolkit_config/.

        Raises
        ------
        OSError
            Config is unable to be used because of missing resources.
        FileNotFoundError
            Requsted config file does not exist.
        """
        home = os.path.expanduser('~')
        configfile = os.path.join(home, f"pfctoolkit_config/{pcc}.json")
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
        """Check that all resources specified in config file are accessible and exist.

        Returns
        -------
        bool
            If True, then all resources specified in config file are accessible and
            exist.
        """
        checks = ["avgr", "fz", "t", "combo", "std", "norm", "chunk_idx"]
        return all(list(map(os.path.exists, [self.config[key] for key in checks])))

    def get(self, key):
        """Get value in config specified by key.

        Parameters
        ----------
        key : str
            Key of value to retrieve from config.

        Returns
        -------
        str
            Value of Key in config.
        """
        return self.config[key]

    def get_config(self):
        """Get config as dict.

        Returns
        -------
        dict
            Configuration as dict.
        """
        return self.config
