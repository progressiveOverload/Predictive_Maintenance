import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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


if __name__ == "__main__":
    main()
