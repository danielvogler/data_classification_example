import streamlit as st
import logging
from dataclassification.config import config
from dataclassification.settings import PROJECT_ROOT


st.set_page_config(layout="wide")

import data
import models
import cost
import summary

### config
config_file = PROJECT_ROOT + '/config/config.ini'
config.initialize_config_vars( config_file )
logging.basicConfig(encoding='utf-8', level=logging.INFO)
logging.info(f'PROJECT_ROOT: {PROJECT_ROOT}')

st.title('Data classification')
PAGES = {
    "Data": data,
    "Models": models,
    "Cost": cost,
    "Summary": summary
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app(config)




