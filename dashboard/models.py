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
from PIL import Image

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
    df_model_summary = pd.read_csv( Path( config.files_dir / 'df_model_summary.csv' ), index_col=0 )
    df_cost_summary = pd.read_csv( Path( config.files_dir / 'df_cost_summary.csv' ), index_col=0 )

    st.dataframe(df_model_summary.sort_values(by=['accuracy_score'], ascending=False))

    st.write('Correct predictions')
    st.image( Image.open( Path(config.fig_dir / 'all_predictions_correct_error_hist.png') ) )

    st.write('Incorrect predictions')
    st.image( Image.open(Path(config.fig_dir / 'all_predictions_incorrect_error_hist.png') ) )