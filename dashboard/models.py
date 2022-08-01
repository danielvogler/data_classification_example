import streamlit as st
import logging
import pandas as pd
import numpy as np
from dataclassification.analytics import Analytics
from dataclassification.config import config
from dataclassification.settings import PROJECT_ROOT
from dataclassification.plotting import Plotting
from dataclassification.utils import Utils
from pathlib import Path
import pickle

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

    st.dataframe(df_summary.sort_values(by=['accuracy_score'], ascending=False))

    st.write('Correct predictions')
    with open(Path( config.fig_dir / 'all_predictions_correct_error_hist.pkl')    , 'rb') as f:
        fig = pickle.load( f )
    st.pyplot(fig, use_container_width=True)

    st.write('Incorrect predictions')
    with open(Path( config.fig_dir / 'all_predictions_incorrect_error_hist.pkl')    , 'rb') as f:
        fig = pickle.load( f )
    st.pyplot(fig, use_container_width=True)

