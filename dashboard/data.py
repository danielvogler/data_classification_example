import streamlit as st
import logging
import pandas as pd
from pathlib import Path
from dataclassification.utils import Utils
from PIL import Image

logging.basicConfig(encoding='utf-8', level=logging.INFO)

def app(config):

    utils = Utils(config)

    st.cache

    sideb = st.sidebar
    df_data = utils.load_data_file()

    ###
    st.subheader('Original data')
    st.write(f'Samples: {df_data.shape[0]}')
    st.write(f'Columns: {df_data.columns.to_list()}')
    st.dataframe(df_data)

    st.write('Data description')
    st.dataframe(df_data.describe())

    st.write('Pairplot')
    st.image( Image.open( Path(config.fig_dir / 'feature_pairplot.png') ) )

    st.write('Correlation heatmap')
    st.image( Image.open( Path(config.fig_dir / 'feature_heatmap.png') ) )

    ###
    st.subheader('Removed data')
    df_data, df_diff = utils.data_preprocessing(df_data)

    cols_to_drop = config.col_drop
    ###
    st.write(f'Dropped columns: {len(config.col_drop)}')
    st.write(f'{config.col_drop}')
    ###
    st.write(f'Dropped rows: {df_diff.shape[0]}')
    st.dataframe(df_diff)

    ###
    st.write(f'Target variable: {config.col_target}')
    df_target = df_data[config.col_target].value_counts()
    df_target = pd.concat([df_data[config.col_target].value_counts(),
                df_data[config.col_target].value_counts(normalize=True).mul(100)],axis=1, keys=('counts','percentage'))
    st.dataframe( df_target )

    st.subheader('Used data')
    st.write(f'Samples: {df_data.shape[0]}')
    st.write(f'Columns: {df_data.columns.to_list()}')
    st.dataframe(df_data)





