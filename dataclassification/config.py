import configparser
import os
from pathlib import Path
import logging
import datetime
from . import settings


class _Config:

    def __init__(self):

        PROJECT_ROOT = settings.PROJECT_ROOT

        if "/.venv/" in PROJECT_ROOT:
            PROJECT_ROOT = PROJECT_ROOT[:PROJECT_ROOT.index("/.venv/")]
        else:
            self.PROJECT_ROOT = Path( PROJECT_ROOT )
            self.PACKAGE_ROOT = Path( Path(self.PROJECT_ROOT) / 'dataclassification/')

        return


    def intialize_project_directories(self):
        ''' create folders (if needed) for model data and figures'''
        Path( self.project_dir ).mkdir(parents=True, exist_ok=True)
        Path( self.log_dir ).mkdir(parents=True, exist_ok=True)
        Path( self.fig_dir ).mkdir(parents=True, exist_ok=True)
        Path( self.files_dir).mkdir(parents=True, exist_ok=True)


    def initialize_config_vars(self, config_file:str = None):
        ''' initialize config variables '''

        config = configparser.ConfigParser()

        if config_file:
            config.read(config_file)
        else:
            config_file = Path(self.PROJECT_ROOT / 'config/config.ini')
            config.read(config_file)

        logging.info(f'Loading config file: ', config_file)

        ### DIRECTORIES
        self.project_dir = Path(self.PROJECT_ROOT / config.get('directories', 'project_dir') )
        self.fig_dir = Path(self.project_dir / config.get('directories', 'fig_dir') )
        self.files_dir = Path(self.project_dir / config.get('directories', 'files_dir') )
        self.log_dir = Path(self.project_dir / config.get('directories', 'log_dir') )

        ### FILES
        self.data_file = Path(self.files_dir / config.get('files', 'data_file') )

        ### data
        self.col_cat = config.get('data', 'col_cat')
        if ',' in self.col_cat:
            self.col_cat = self.col_cat.split(',')
        self.col_target = config.get('data', 'col_target')
        self.col_drop = config.get('data', 'col_drop')
        if ',' in self.col_drop:
            self.col_drop = self.col_drop.split(',')

        ### model
        self.test_size = float( config.get('model', 'test_size') )
        self.all_model_names = config.get('model', 'all_model_names').split(',')

        ### cost
        self.buy_cost = float( config.get('cost', 'buy_cost') )
        self.sell_prices = [float(x) for x in config.get('cost', 'sell_prices').split(',')]
        self.sell_categories = config.get('cost', 'sell_categories').split(',')
        self.sell_dict = {self.sell_categories[i]: self.sell_prices[i] for i in range(len(self.sell_prices))}
        self.weekly_packages = float( config.get('cost', 'weekly_packages') )
        self.weekly_sorting_cost = float( config.get('cost', 'weekly_sorting_cost') )

        ### plotting
        self.font_axis = config.get('plotting', 'font_axis')
        self.font_ticks = config.get('plotting', 'font_ticks')
        self.font_label = config.get('plotting', 'font_label')
        self.plot_lw = config.get('plotting', 'plot_lw')

        self.intialize_project_directories()

        return


    def __getattr__(self, name):
        try:
            return self.config[name]
        except:
            raise ValueError(f'Config variable not found: {name}')


config = _Config()