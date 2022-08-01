import streamlit as st
import logging
import pandas as pd
import numpy as np
from dataclassification.analytics import Analytics
from dataclassification.config import config
from dataclassification.settings import PROJECT_ROOT
from dataclassification.plotting import Plotting
from dataclassification.utils import Utils


### config
config_file = PROJECT_ROOT + '/config/config.ini'
config.initialize_config_vars( config_file )
logging.basicConfig(encoding='utf-8', level=logging.INFO)
logging.info(f'PROJECT_ROOT: {PROJECT_ROOT}')

### init
ana = Analytics(config)
plts = Plotting(config)
utils = Utils(config)

def app():

    st.cache

    sideb = st.sidebar
    st.subheader('Cost evaluation')
    all_model_names = config.all_model_names
    df_cost_sum = ana.all_cost_analyses(all_model_names)
    st.dataframe(df_cost_sum)





