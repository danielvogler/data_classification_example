import streamlit as st
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(encoding='utf-8', level=logging.INFO)

def app(config):
    sideb = st.sidebar

    st.subheader('Cost evaluation')
    df_cost_summary = pd.read_csv( Path( config.files_dir / 'df_cost_summary.csv' ), index_col=0 )
    st.dataframe(df_cost_summary.sort_values(by=df_cost_summary.columns[0], ascending=True))





