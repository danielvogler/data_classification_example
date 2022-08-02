""" Example data classification

"""
import logging
import sys

from dataclassification.config import config
from dataclassification.settings import PROJECT_ROOT
from dataclassification.plotting import Plotting
from dataclassification.analytics import Analytics
from dataclassification.utils import Utils


### config
config_file = PROJECT_ROOT + '/config/config.ini'
config.initialize_config_vars( config_file )
logging.getLogger().setLevel(logging.INFO)
logging.info(f'PROJECT_ROOT: {PROJECT_ROOT}')

### init
plts = Plotting(config)
utils = Utils(config)
ana = Analytics(config)

### files
df_data = utils.load_data_file()
df_data, df_diff = utils.data_preprocessing(df_data)

# fig = plts.data_pairplot(df_data)
# fig = plts.data_heatmap(df_data)

df_model_summary, df_cost_summary = ana.all_models(df_data, config.all_model_names)
logging.info(f'Model summary:\n{df_model_summary}')
logging.info(f'Cost summary:\n{df_cost_summary}')
