import pandas as pd
import ydata_profiling as pdpf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Define the new path to your dataset
    new_dataset_path = 'D:/Projects/predictive_maintenance/dataset_1.csv'

    # Read the dataset from the new path
    df = pd.read_csv(new_dataset_path)

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


if __name__ == "__main__":
    main()
