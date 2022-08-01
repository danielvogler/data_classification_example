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

    sideb = st.sidebar

    st.subheader('Model performance')
    df_data = utils.load_data_file()
    df_data, df_diff = utils.data_preprocessing(df_data)

    all_model_names = config.all_model_names
    selected_models = sideb.multiselect(
     'Model names',
     all_model_names,
     all_model_names)
    df_summary = ana.all_models(df_data, selected_models)

    st.dataframe(df_summary[['accuracy_score']])


