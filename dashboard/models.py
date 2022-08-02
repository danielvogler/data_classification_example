import streamlit as st
import logging
import pandas as pd
from pathlib import Path
from PIL import Image

logging.basicConfig(encoding='utf-8', level=logging.INFO)

def app(config):

    sideb = st.sidebar

    st.subheader('Model performance')
    df_model_summary = pd.read_csv( Path( config.files_dir / 'df_model_summary.csv' ), index_col=0 )
    st.dataframe(df_model_summary.sort_values(by=['accuracy_score'], ascending=False))

    st.write('Correct predictions')
    st.image( Image.open( Path(config.fig_dir / 'all_predictions_correct_error_hist.png') ) )

    st.write('Incorrect predictions')
    st.image( Image.open(Path(config.fig_dir / 'all_predictions_incorrect_error_hist.png') ) )

    st.write('Feature importance')
    df_fi_summary = pd.read_csv( Path( config.files_dir / 'df_feature_importance_summary.csv' ), index_col=0 )
    st.dataframe(df_fi_summary)