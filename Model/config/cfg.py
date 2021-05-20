import os
import json
import torch
import argparse
from config.logger import Logger

class Config(object):
    def __init__(self, args=None, logger=None):
        self.logger = logger
        self.config = vars(args) if args is not None else args
        
    def save_config(self, path):
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        if self.logger is None:
            print("Config saved to file {}".format(path))
        else:
            self.logger.debug("Config saved to file {}".format(path))

    def load_config(self, path, verbose=True):
        with open(path) as f:
            self.config = json.load(f)
        if self.logger is None:
            print("Config loaded from file {}".format(path))
        else:
            self.logger.debug("Config loaded from file {}".format(path))

    def print_config(self):
        debug = "Running with the following configs:\n"
        for k,v in self.config.items():
            debug += "\t{} : {}\n".format(k, str(v))
        if self.logger is None:
            print("\n" + debug + "\n")  
        else:          
            self.logger.debug("\n" + debug + "\n")