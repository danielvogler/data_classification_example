import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import TypeVar, List
import pickle
from .utils import Utils
import numpy as np

class Plotting:

    def __init__(self, config) -> None:
        self.config = config
        return


    def data_pairplot(
        self,
        df: pd.DataFrame
        ) -> TypeVar:
        ''' pairplot of dataframe
        :param df: dataframe '''
        fig = sns.pairplot(df, diag_kind='kde', hue=self.config.col_target)
        fig.savefig( Path( self.config.fig_dir / 'feature_pairplot.png'), bbox_inches='tight' )

        with open( Path( self.config.fig_dir / 'feature_pairplot.pkl'), 'wb') as f:
            pickle.dump(fig, f)

        return fig


    def data_heatmap(
        self,
        df: pd.DataFrame
        ) -> TypeVar:
        ''' heatmap of dataframe
        :param df: dataframe '''
        fig = plt.figure(figsize=(15,10))
        df, category_dict = Utils(self.config).map_categorical_values(df, self.config.col_cat)
        sns.set(font_scale=1.3)
        sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap='RdBu_r')
        fig.savefig( Path( self.config.fig_dir / 'feature_heatmap.png'), bbox_inches='tight' )

        with open( Path( self.config.fig_dir / 'feature_heatmap.pkl'), 'wb') as f:
            pickle.dump(fig, f)

        return fig


    def categorization_error_hist(
        self,
        model_name: str,
        y_test,
        prediction
        ) -> None:

        prediction_comparison = (y_test==prediction)

        prediction_correct = y_test[prediction_comparison]
        prediction_incorrect = y_test[~prediction_comparison]

        plt.figure(figsize=(20, 15))
        plt.hist([prediction_correct, prediction_incorrect], stacked="True", label=['True', 'False'])
        plt.title('Predictions correct', fontsize=self.config.font_axis)
        plt.ylabel('Frequency [-]', fontsize=self.config.font_axis)
        plt.xlabel('Category [-]', fontsize=self.config.font_axis)
        plt.xticks(fontsize=self.config.font_ticks)
        plt.yticks(fontsize=self.config.font_ticks)
        plt.legend(loc='upper right', fontsize=self.config.font_axis)
        plt.savefig( Path(self.config.fig_dir / f'{model_name}_predictions_error_hist.png'), bbox_inches='tight' )

        return


    def all_categorization_error_hist(
        self,
        all_model_names: List[str]
        ) -> None:


        df_data = Utils(self.config).load_data_file()

        df_comp_corr = pd.DataFrame()
        df_comp_incorr = pd.DataFrame()

        df_comp_corr.index = df_data[self.config.col_target].unique()
        df_comp_incorr.index = df_data[self.config.col_target].unique()

        for m in all_model_names:
            df_model = Utils(self.config).load_prediction_and_target_values(m)
            y_test, prediction = df_model['y_test'], df_model['prediction']
            prediction_comparison = (y_test==prediction)

            prediction_correct = y_test[prediction_comparison]
            prediction_incorrect = y_test[~prediction_comparison]

            correct_values_counts = prediction_correct.value_counts()
            incorrect_values_counts = prediction_incorrect.value_counts()

            df_comp_corr = pd.merge(df_comp_corr, correct_values_counts, how='left', left_index=True, right_index=True)
            df_comp_corr = df_comp_corr.rename(columns={"y_test": m})

            df_comp_incorr = pd.merge(df_comp_incorr, incorrect_values_counts, how='left', left_index=True, right_index=True)
            df_comp_incorr = df_comp_incorr.rename(columns={"y_test": m})

        fig = plt.figure(figsize=(20, 15))
        df_comp_corr.plot.bar()
        plt.title('Predictions correct', fontsize=self.config.font_axis)
        plt.ylabel('Frequency [-]', fontsize=self.config.font_axis)
        plt.xlabel('Category [-]', fontsize=self.config.font_axis)
        plt.xticks(fontsize=self.config.font_ticks)
        plt.yticks(fontsize=self.config.font_ticks)
        plt.legend(loc='upper right', fontsize=self.config.font_axis)
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.savefig( Path(self.config.fig_dir / f'all_predictions_correct_error_hist.png'), bbox_inches='tight' )

        with open( Path( self.config.fig_dir / 'all_predictions_correct_error_hist.pkl'), 'wb') as f:
            pickle.dump(fig, f)

        fig = plt.figure(figsize=(20, 15))
        df_comp_incorr.plot.bar()
        plt.title('Predictions incorrect', fontsize=self.config.font_axis)
        plt.ylabel('Frequency [-]', fontsize=self.config.font_axis)
        plt.xlabel('Category [-]', fontsize=self.config.font_axis)
        plt.xticks(fontsize=self.config.font_ticks)
        plt.yticks(fontsize=self.config.font_ticks)
        plt.legend(loc='upper right', fontsize=self.config.font_axis)
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.savefig( Path(self.config.fig_dir / f'all_predictions_incorrect_error_hist.png'), bbox_inches='tight' )

        with open( Path( self.config.fig_dir / 'all_predictions_incorrect_error_hist.pkl'), 'wb') as f:
            pickle.dump(fig, f)

        return