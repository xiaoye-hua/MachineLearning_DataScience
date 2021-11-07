import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def count_plot(df: pd.DataFrame, col: str, xytext=(0, 0), show_details=True) -> None:
    '''
    custom count plot
    Args:
        df:
        col:
        xytext:

    Returns:

    '''
    ax = sns.countplot(data=df, x=col)
    if show_details:
        for bar in ax.patches:
            ax.annotate('%{:.2f}\n{:.0f}'.format(100*bar.get_height()/len(df),bar.get_height()), (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                        size=11, xytext=xytext,
                        textcoords='offset points')
    plt.show()


def labels(ax, df, xytext=(0, 0)):
    for bar in ax.patches:
        ax.annotate('%{:.2f}\n{:.0f}'.format(100*bar.get_height()/len(df),bar.get_height()), (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                    size=11, xytext=xytext,
                    textcoords='offset points')


def cate_features_plot(df, col, target, target_binary=True, figsize=(20,6)):
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)

    plt.subplot(121)
    if target_binary:
        tmp = round(pd.crosstab(df[col], df[target], normalize='index'), 2)
        tmp = tmp.reset_index()
        # tmp.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)

    ax[0] = sns.countplot(x=col, data=df, hue=target,
                          order=np.sort(df[col].dropna().unique()),
                          )
    ax[0].tick_params(axis='x', rotation=90)
    labels(ax[0], df[col].dropna(), (0, 0))
    if target_binary:
        ax_twin = ax[0].twinx()
        # sns.set(rc={"lines.linewidth": 0.7})
        ax_twin = sns.pointplot(x=col, y=1, data=tmp, color='black', legend=False,
                                order=np.sort(df[col].dropna().unique()),
                                linewidth=0.1)

    ax[0].grid()

    plt.subplot(122)
    ax[1] = sns.countplot(x=df[col].dropna(),
                          order=np.sort(df[col].dropna().unique()),
                          )
    ax[1].tick_params(axis='x', rotation=90)
    labels(ax[1], df[col].dropna())
    plt.show()

def get_col_type_dist(df: pd.DataFrame) -> None:
        """
        get distribution of dataframe column for feature creating
        Args:
            df:
        Returns:
        """
        types = pd.DataFrame(df.dtypes)
        type_dict = {}
        for row in types.iterrows():
            try:
                type_dict[row[1][0]].append(row[0])
            except:
                type_dict[row[1][0]] = [row[0]]
        for key, value in type_dict.items():
            print('*' * 20)
            print(f'Column Type: {key}')
            print(f'Column number: {len(type_dict[key])}')
        print('String type columns:')
        print(type_dict[np.dtype('O')])