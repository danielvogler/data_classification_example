import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import TypeVar
import pickle
from .utils import Utils

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
        plt.hist([prediction_correct, prediction_incorrect], label=['True', 'False'])
        plt.title('Predictions correct', fontsize=self.config.font_axis)
        plt.ylabel('Frequency [-]', fontsize=self.config.font_axis)
        plt.xlabel('Category [-]', fontsize=self.config.font_axis)
        plt.xticks(fontsize=self.config.font_ticks)
        plt.yticks(fontsize=self.config.font_ticks)
        plt.legend(loc='upper right', fontsize=self.config.font_axis)
        plt.savefig( Path(self.config.fig_dir / f'{model_name}_predictions_error_hist.png'), bbox_inches='tight' )

        return