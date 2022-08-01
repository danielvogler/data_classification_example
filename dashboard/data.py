import streamlit as st
import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
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
    df_data = utils.load_data_file()
    df_data, df_diff = utils.data_preprocessing(df_data)

    ###
    st.subheader('Original data')
    st.dataframe(df_data)

    st.write('Data description')
    st.dataframe(df_data.describe())

    st.write('Pairplot')
    with open(Path( config.fig_dir / 'feature_pairplot.pkl') , 'rb') as f:
        fig = pickle.load( f )
    st.pyplot(fig, use_container_width=True)

    st.write('Heatmap')
    with open(Path( config.fig_dir / 'feature_heatmap.pkl') , 'rb') as f:
        fig = pickle.load( f )
    st.pyplot(fig, use_container_width=True)

    ###
    st.subheader('Removed data')
    cols_to_drop = config.col_drop
    ###
    st.write(f'Dropped columns: {len(config.col_drop)}')
    st.write(f'{config.col_drop}')
    ###
    st.write(f'Dropped rows: {df_diff.shape[0]}')
    st.dataframe(df_diff)

    ###
    st.write(f'Target variable:')
    df_target = df_data[config.col_target].value_counts()
    df_target = pd.concat([df_data[config.col_target].value_counts(),
                df_data[config.col_target].value_counts(normalize=True).mul(100)],axis=1, keys=('counts','percentage'))
    st.dataframe( df_target )





