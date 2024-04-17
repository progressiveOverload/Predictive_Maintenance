import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

warnings.filterwarnings("ignore", message=".*invalid escape sequence.*", category=SyntaxWarning)
warnings.filterwarnings("ignore", message=".*use_inf_as_na option is deprecated.*", category=FutureWarning)


def main():
    # Set the use_inf_as_na option to False
    pd.set_option('mode.use_inf_as_na', False)

    # Define the new path to your dataset
    new_dataset_path = 'D:/Projects/predictive_maintenance/dataset_1.csv'

    # Read the dataset from the new path
    df = pd.read_csv(new_dataset_path)

    # Convert inf values to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # total number of rows and columns
    print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

    df.head(3)

    # rename dataset columns
    df.rename(columns={'Air temperature [K]': 'Air temperature',
                       'Process temperature [K]': 'Process temperature',
                       'Rotational speed [rpm]': 'Rotational speed',
                       'Torque [Nm]': 'Torque',
                       'Tool wear [min]': 'Tool wear'},
              inplace=True)

    df.drop(['Product ID', 'UDI'], axis=1, inplace=True)

    df.head(2)

    df.info()

    # overall descriptive information on numerical attributes
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric.describe().transpose()

    # overall descriptive information on categorical attributes
    df_categorical = df.select_dtypes(include=[np.object_])
    df_categorical.describe().transpose()

    fig, ax = plt.subplots(3, 4, figsize=(25, 20))

    for i, col in enumerate(df.columns):
        sns.histplot(df[col], ax=ax[i // 4][i % 4])

    plt.show()

    # Reset the use_inf_as_na option to its default value
    pd.reset_option('mode.use_inf_as_na')

    df_failures = df.loc[:, ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
    # Calculate the sum of the values in each row
    rows_sum = df_failures.sum(axis=1)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.countplot(x=rows_sum, ax=ax)
    for patch in ax.patches:
        ax.annotate(str(patch.get_height()), (patch.get_x() + patch.get_width() / 2, patch.get_height()), ha='center',
                    va='bottom')

    ax.set_title('Number of failure types per record')

    df['Machine failure'] = 0

    df.loc[df['TWF'] == 1, 'Machine failure'] = 1
    df.loc[df['HDF'] == 1, 'Machine failure'] = 1
    df.loc[df['PWF'] == 1, 'Machine failure'] = 1
    df.loc[df['OSF'] == 1, 'Machine failure'] = 1
    df.loc[df['RNF'] == 1, 'Machine failure'] = 1

    # drop individual failure types
    df.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)

    failure_types = df.loc[:, ['Machine failure']]

    rows_sum = failure_types.sum(axis=1)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.countplot(x=rows_sum, ax=ax)
    for patch in ax.patches:
        ax.annotate(str(patch.get_height()), (patch.get_x() + patch.get_width() / 2, patch.get_height()), ha='center',
                    va='bottom')
        ax.set_title('Count of different failure types')

    df['Power'] = df[['Rotational speed', 'Torque']].product(axis=1)

    sns.histplot(df['Power'])

    # convert Type attribute into numbers, such that L = 0, M = 1, and H = 2
    df['Type'].replace('L', 0, inplace=True)
    df['Type'].replace('M', 1, inplace=True)
    df['Type'].replace('H', 2, inplace=True)

    # turn all columns into float for easier processing later
    for column in df.columns:
        df[column] = df[column].astype(float)

    print(df.dtypes)

    # List of columns to exclude from normalization and winsorization
    excluded_columns = ['Type', 'Machine failure']

    for col in df.columns:
        if col not in excluded_columns:
            # calculate the IQR (interquartile range)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] <= (Q1 - 1.5 * IQR)) | (df[col] >= (Q3 + 1.5 * IQR))]
            if not outliers.empty:
                # df.loc[outliers.index, col] = winsorize(outliers[col], limits=[0.08, 0.08])
                df.drop(outliers.index, inplace=True)

    # create the LOF model
    model = LocalOutlierFactor(n_neighbors=5)

    # use the model to predict the outlier scores for each row
    scores = model.fit_predict(df)

    # identify the outlier rows (those with a negative score) and remove them
    outliers = df[scores == -1]
    if not outliers.empty:
        df.drop(outliers.index, inplace=True)

    print(df.shape)

    # Iterate over the columns in the dataframe
    for col in df.columns:
        if col not in excluded_columns:
            # Normalize the values in the column
            df[col] = zscore(df[col])


if __name__ == "__main__":
    main()
