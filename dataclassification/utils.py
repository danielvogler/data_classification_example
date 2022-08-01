
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


    def model_preprocessing(
        self,
        df: pd.DataFrame
        ) -> Tuple:
        """ Preprocess model

        Args:
            df (pd.DataFrame): data

        Returns:
            Tuple: Data for split train/test and feature/target
        """
        logging.info('Model preprocessing')

        X_train, X_test = train_test_split(df, test_size=self.config.test_size, shuffle=True)

        X_train, y_train = self.split_feature_target(X_train, self.config.col_target)
        X_test, y_test = self.split_feature_target(X_test, self.config.col_target)

        scaler, X_train, X_test = self.scale_X(X_train, X_test)

        return X_train, X_test, y_train, y_test, scaler


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


    @staticmethod
    def prediction_error(
        prediction,
        y_test) -> Tuple:
        """ Compute a few prediction errors

        Args:
            prediction (_type_): target prediction
            y_test (_type_): target values

        Returns:
            Tuple: error metrics
        """
        logging.info('Error calculation:')

        if isinstance(y_test[0], (int, float, np.int8, np.int32, np.int64)):
            logging.info('Numeric target')
            MAE = round( mean_absolute_error(y_test , prediction), 3)
            MAPE = round( mean_absolute_percentage_error(y_test , prediction), 3)
            r2 = round( r2_score(y_test, prediction), 3)

        elif isinstance(y_test[0], str):
            logging.info('Categorical target')
            MAE = None
            MAPE = None
            r2 = None

        else:
            raise Exception('Incorrect file type')

        acc_score = round( accuracy_score(y_test, prediction), 3)

        logging.info(f'MAE: {MAE}')
        logging.info(f'MAPE: {MAPE}')
        logging.info(f'R2: {r2}')
        logging.info(f'accuracy_score: {acc_score}')

        return MAE, MAPE, r2, acc_score


    def save_prediction_and_target_values(
        self,
        model_name: str,
        y_test,
        prediction
        ) -> None:
        """ Save mapping of y_test -> prediction

        Args:
            model_name (str): model name
            y_test (_type_): target values for testing
            prediction (_type_): predicted target values
        """
        file_name = f'{model_name}_ytest_prediction_comparison.csv'
        logging.info(f'Save ytest/prediction comparison to: {file_name}')

        df = pd.DataFrame({"y_test" : y_test, "prediction" : prediction})
        df.to_csv( Path(self.config.files_dir / f'{file_name}'), index=False)

        return


    def load_prediction_and_target_values(
        self,
        model_name: str
        ) -> None:
        """ Load mapping of y_test -> prediction

        Args:
            model_name (str): model name

        Returns:
            pd.DataFrame: Data DF
        """
        file_name = f'{model_name}_ytest_prediction_comparison.csv'
        logging.info(f'Load error calculation to: {file_name}')

        df = pd.read_csv( Path(self.config.files_dir / f'{file_name}') )

        return df