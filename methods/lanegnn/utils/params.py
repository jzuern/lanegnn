import argparse
import yaml
import os
import sys


class ParamNamespace:
    """
    One dictionary entity holding parameters converted to a class.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def overwrite(self, args: argparse.Namespace):
        for k, v in vars(args).items():
            if k in self.__dict__.keys() and v is not None:
                self.__dict__[k] = v


class ParamLib:
    """
    Combines multiple ParamNamespace entities in one larger class.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path

        # Create all parameter dictionaries
        self.main = ParamNamespace()
        self.paths = ParamNamespace()
        self.preprocessing = ParamNamespace()
        self.model = ParamNamespace()
        self.driving = ParamNamespace()

        # Load config file with parametrization, create paths and do sys.path.inserts
        self.load_config_file(self.config_path)
        #self.create_dir_structure()
        self.add_system_paths()

    def load_config_file(self, path: str):
        """
        Loads a config YAML file and sets the different dictionaries.
        Args:
            path: path to some configuration file in yaml format

        Returns:
        """

        with open(path, 'r') as stream:
            try:
                config_file = yaml.safe_load(stream)
            except yaml.YAMLError as exception:
                print(exception)

        # Copy yaml content to the different dictionaries.
        vars(self.main).update(config_file['main'])
        vars(self.paths).update(config_file['paths'])
        vars(self.preprocessing).update(config_file['preprocessing'])
        vars(self.model).update(config_file['model'])
        vars(self.driving).update(config_file['driving'])

        # # Set some secondary paths that are important
        # if self.main.dataset == "paloalto":
        #     pass
        #     # print("using palo alto")
        #     # paths to preprocessed data
        #     #self.paths.preprocessed_data = os.path.join(self.paths.home, self.main.dataset, 'preprocessed/')
        #
        # else:
        #     raise NotImplementedError

    def create_dir_structure(self):
        """
        Loops through the paths dictionary in order to create
        the paths if they do not exist.
        Args:
            paths_dict: some para

        Returns:
            -
        """
        for name, path in vars(self.paths).items():
            # exclude all paths to files
            if len(path.split('.')) == 1:
                if not os.path.exists(path):
                    os.makedirs(path)

    def add_system_paths(self):
        """
        Loops through the paths dictionary in order to create
        the paths if they do not exist.
        Args:
            paths_dict: some para

        Returns:
            -
        """
        sys.path.insert(0, self.paths.package)
