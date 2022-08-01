
import logging
from os import stat
import pandas as pd
from typing import TypeVar, Tuple, Dict, List
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class Utils:

    def __init__(
        self,
        config
        ) -> None:
        """ Initialize config variables

        Args:
            config (_type_): config object
        """
        self.config = config

        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)

        return


    def load_data_file(
        self
        ) -> pd.DataFrame:
        """  Load example data file

        Returns:
            pd.DataFrame: Data DF
        """
        logging.info(f'Loading data file: {self.config.data_file}')
        df = pd.read_csv(self.config.data_file, usecols=lambda c: not c.startswith('Unnamed:'))

        return df


    @staticmethod
    def map_categorical_values(
        df,
        col_cat: str
        ) -> Tuple[pd.DataFrame, Dict]:
        """ Map categorical categories

        Args:
            df (_type_): data
            col_cat (str): categorical categories

        Returns:
            Tuple[pd.DataFrame, Dict]: dataframe with mapped categories and corresponding Dict
        """
        logging.info('Mapping categorical values')

        df_new = df.copy(deep=True)
        categories = df_new[col_cat].astype('category')
        category_dict = dict( enumerate( categories.cat.categories ) )
        df_new[col_cat] = categories.cat.codes

        return df_new, category_dict


    def data_preprocessing(
        self,
        df: pd.DataFrame
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """_summary_

        Args:
            df (pd.DataFrame): DF in

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: clean DF, removed part of DF
        """
        logging.info('Data preprocessing')
        logging.info(f'Original DF size: {df.shape}')

        df_new = df[df.notnull()]
        df_new = df_new.dropna()

        df_diff = df.merge(df_new.drop_duplicates(),
                   how='left', indicator=True)
        df_diff = df_diff[df_diff['_merge']=='left_only']

        df_new = df_new.drop(columns=self.config.col_drop)

        logging.info(f'Resulting DF size: {df_new.shape}')

        return df_new, df_diff


    @staticmethod
    def scale_X(
        X_train,
        X_test
        ) -> Tuple:
        """ Scale features

        Args:
            X_train (_type_): feature test values
            X_test (_type_): feature test values

        Returns:
            Tuple: Used scaler, train data and test data
        """
        logging.info('Scaling X')

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return scaler, X_train_scaled, X_test_scaled


    @staticmethod
    def split_feature_target(
        X: pd.DataFrame,
        col_target: str
        ) -> Tuple[pd.DataFrame, pd.Series]:
        """ Split feature and target

        Args:
            X (pd.DataFrame): dataframe
            col_target (str): target column of dataframe

        Returns:
            Tuple: feature and target
        """
        logging.info('Splitting feature and target')

        y = X[col_target].to_numpy()
        X.drop([col_target], inplace=True, axis=1)

        return X, y

