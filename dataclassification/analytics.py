import logging
from typing import Dict, Tuple, List
import pandas as pd
from dataclassification.utils import Utils
from dataclassification.plotting import Plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

class Analytics:

    def __init__(self, config) -> None:
        self.config = config
        return


    def all_models(
        self,
        df_data: pd.DataFrame,
        all_model_names: List
        ) -> pd.DataFrame:
        """ Perform modeling for a range of models

        Args:
            df_data (pd.DataFrame): data to model
            all_model_names (List): List of model names

        Returns:
            pd.DataFrame: Summary of all model results
        """
        logging.info(f'Modeling: {all_model_names}')

        df_summary = pd.DataFrame(columns=['model_name','MAE','MAPE','r2','accuracy_score','cv_score'], index=all_model_names)
        df_summary.set_index('model_name', inplace = True)
        df_summary = df_summary.dropna()

        X_train, X_test, y_train, y_test, scaler = Utils(self.config).model_preprocessing(df_data)

        for m in all_model_names:
            MAE, MAPE, r2, accuracy_score, cv_score = self.model_data(X_train,
                                                            X_test,
                                                            y_train,
                                                            y_test,
                                                            m)

            df_summary.loc[m] = pd.Series({'MAE':MAE, 'MAPE':MAPE, 'r2':r2, 'accuracy_score':accuracy_score, 'cv_score':cv_score})

        return df_summary


    def model_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str = 'RF'
        ) -> Tuple:
        """ Model data

        Args:
            df (pd.DataFrame): data
            model_name (str, optional): Name of model to use. Defaults to 'RF'.

        Returns:
            Tuple: Metrics
        """
        logging.info(f'{model_name} modeling')

        # df, category_dict = Utils.map_categorical_values(df, self.config.col_cat)

        if model_name == 'RF':
            logging.info(f' -- RF modeling --')
            model = RandomForestClassifier()

        elif model_name == 'DT':
            logging.info(f' -- DT modeling --')
            model = DecisionTreeClassifier()

        elif model_name == 'SVM':
            logging.info(f' -- SVM modeling --')
            model = svm.SVC()

        elif model_name == 'logReg':
            logging.info(f' -- log. Reg. modeling --')
            model = LogisticRegression()

        else:
            logging.raiseExceptions

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        MAE, MAPE, r2, accuracy_score = Utils.prediction_error(prediction, y_test)

        cv_score = cross_val_score(estimator = model, X = X_train, y = y_train.ravel(), cv = 10).mean()

        # ### categorize
        # y_test_cat = np.vectorize(category_dict.get)(y_test)
        # prediction_cat = np.vectorize(category_dict.get)(prediction)

        Plotting(self.config).categorization_error_hist(model_name, y_test, prediction)
        Utils(self.config).save_prediction_and_target_values(model_name, y_test, prediction)

        return MAE, MAPE, r2, accuracy_score, cv_score


    def all_cost_analyses(
        self,
        all_model_names: List
        ) -> pd.DataFrame:

        df_cost_summary = pd.DataFrame(columns=['model_name','weekly_reimbursements_sum'], index=all_model_names)
        df_cost_summary.set_index('model_name', inplace = True)
        df_cost_summary = df_cost_summary.dropna()

        for m in all_model_names:
            weekly_reimbursement_sum = self.cost_analysis( m )
            df_cost_summary.loc[m] = pd.Series({'weekly_reimbursements_sum': weekly_reimbursement_sum})

        return df_cost_summary


    def cost_analysis(
        self,
        model_name: str
        ) -> float:

        weekly_purchasing_cost = self.config.weekly_packages * self.config.buy_cost
        df_comparison = Utils(self.config).load_prediction_and_target_values(model_name)

        df_incorrect = df_comparison[ (df_comparison['y_test'] != df_comparison['prediction']) ].copy(deep=True)
        df_incorrect['to_reimburse'] = df_incorrect['prediction'].replace( self.config.sell_dict )

        test_reimbursement_sum = df_incorrect['to_reimburse'].sum()
        weekly_reimbursement_sum = round( test_reimbursement_sum * self.config.weekly_packages / df_comparison.shape[0], 2)

        logging.info(f'Model ({model_name}) weekly reimbursements: {weekly_reimbursement_sum}')

        return weekly_reimbursement_sum
