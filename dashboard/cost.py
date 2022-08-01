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

    st.subheader('Cost evaluation')
    df_cost_summary = pd.read_csv( Path( config.files_dir / 'df_cost_summary.csv' ), index_col=0 )
    st.dataframe(df_cost_summary.sort_values(by=df_cost_summary.columns[0], ascending=True))





