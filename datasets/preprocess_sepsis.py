"""
Data processing of sepsis dataset.
"""

import random
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset

FEATURES = ['HR', 'Temp', 'Resp', 'MAP', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Age', 'ICULOS']


def get_train_test_loaders_encoding(data_path, train_ratio, ts_pad_lenght, no_sepsis_only):
    """
    Encapsulating function to build dataloaders for encoding (no labels).
    """
    # Build Pytorch datasets for train and test
    dataframe, train_ids, test_ids = get_ids_train_test_sepsis(data_path, train_ratio, no_sepsis_only)
    train_dataset, num_features, clean_train_dataset = get_dataset_encoding(dataframe, train_ids, ts_pad_lenght)
    print(f"There are {len(train_dataset)} patients in the train dataset.")
    test_dataset, _, clean_test_dataset = get_dataset_encoding(dataframe, test_ids, ts_pad_lenght)
    print(f"There are {len(test_dataset)} patients in the test dataset.")

    return train_dataset, test_dataset, num_features, clean_train_dataset, clean_test_dataset


def get_ids_train_test_sepsis(data_path, train_ratio, no_sepsis_only):
    """
    Get dataframe and train and test patient ids with given train ratio of each in train.
    """

    dataframe = pd.read_csv(data_path)

    # Remove outliers
    dataframe = replace_outliers_by_nan(dataframe)

    if no_sepsis_only:
        no_sepsis_df = dataframe[dataframe['SepsisLabel'] == 1]  # change to 0
        no_sepsis_ids = no_sepsis_df['ID'].unique()

        random.seed(42)
        train_no_sepsis_idxs = random.sample(range(len(no_sepsis_ids)), int(len(no_sepsis_ids) * train_ratio))

        train_ids = [no_sepsis_ids[i] for i in train_no_sepsis_idxs]
        test_ids = list(set(no_sepsis_ids) - set(train_ids))
    else:
        sepsis_df = dataframe[dataframe['SepsisLabel'] == 1]
        normal_ids = list(set(dataframe['ID'].unique()) - set(sepsis_df['ID'].unique()))
        sepsis_ids = list(sepsis_df['ID'].unique())

        random.seed(42)
        train_normal_idxs = random.sample(range(len(normal_ids)), int(len(normal_ids) * train_ratio))
        random.seed(52)
        train_sepsis_idxs = random.sample(range(len(sepsis_ids)), int(len(sepsis_ids) * train_ratio))

        train_ids = [normal_ids[i] for i in train_normal_idxs] + [sepsis_ids[i] for i in train_sepsis_idxs]
        test_ids = [normal_ids[i] for i in set(range(len(normal_ids))) - set(train_normal_idxs)] + \
                   [sepsis_ids[i] for i in set(range(len(sepsis_ids))) - set(train_sepsis_idxs)]

    # Normalize whole dataframe with train values
    train_df = dataframe[dataframe['ID'].isin(train_ids)]
    train_means = train_df[FEATURES].mean()
    train_stds = train_df[FEATURES].std()
    dataframe[FEATURES] = (dataframe[FEATURES] - train_means[FEATURES]) / (train_stds[FEATURES] + 1e-4)

    return dataframe, train_ids, test_ids


def replace_outliers_by_nan(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Find outliers (defined with IQR) in the dataframe and replace them with NaN values.

    Inputs:
    * dataframe (pd.DataFrame): Dataframe to remove outliers from.

    Return:
    * Dataframe where outliers are replaced with NaN values.
    """
    # compute 5th and 95th quantiles
    q75 = dataframe.quantile(q=0.95, numeric_only=True)  # can also be 0.75 and 0.25
    q25 = dataframe.quantile(q=0.05, numeric_only=True)
    for col in FEATURES:
        if col == "ICULOS" or col == "Age":
            continue
        else:
            # compute "fences" with the interquantile range which define an outlier
            lower_fence = q25[col] - (q75[col] - q25[col]) * 3
            upper_fence = q75[col] + (q75[col] - q25[col]) * 3

            # get the values of the current column for the current id group
            values = dataframe[col].values

            # loop over each date
            for i, _ in enumerate(values):
                # if the current value is an outlier, replace it with NaN
                if values[i] > upper_fence or values[i] < lower_fence:
                    values[i] = np.nan
            # update the original dataframe with the new values
            dataframe[col] = values

    return dataframe


class TimeSeriesDataset(Dataset):
    """
    Generic Pytorch dataset for time series.
    """

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def pad_and_cut(dataframe, id_list, ts_pad_length):
    print("Start padding and cutting")
    dataframes_list = []

    dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.fillna(method='bfill')

    age_dict = dataframe.groupby('ID')['Age'].first().to_dict()

    for pat_id in id_list:
        pat_df = dataframe[dataframe['ID'] == pat_id].copy()
        pat_df = pat_df.interpolate(limit_direction="both")  # Interpolate missing data roughly

        # Calculate the number of rows needed for padding at the beginning
        num_padding_rows = max(0, ts_pad_length - len(pat_df))

        # Cut the data to make it of size ts_pad_length and preserve the end
        if len(pat_df) > ts_pad_length:
            pat_df = pat_df.iloc[len(pat_df) - ts_pad_length:]

        if num_padding_rows > 0:
            padding_df = pd.DataFrame(
                {'HR': 0, 'Temp': 0, 'Resp': 0, 'MAP': 0, 'Creatinine': 0, 'Bilirubin_direct': 0,
                 'Glucose': 0, 'Lactate': 0, 'ICULOS': 0, "SepsisLabel": 0, 'ID': pat_id, 'Age': age_dict[pat_id]},
                index=range(num_padding_rows))
            pat_df = pd.concat([padding_df, pat_df], ignore_index=True)

        dataframes_list.append(pat_df)

    clean_df = pd.concat(dataframes_list)

    return clean_df


def get_dataset_encoding(dataframe, id_list, ts_pad_lenght):
    """
    Build padded dataset with sequences of the patients in the given id_list.
    Dataset for instance classification, ie one patient has one label for the whole time serie
    """
    data_list = []
    label_list = []

    dataframe = pad_and_cut(dataframe, id_list, ts_pad_lenght)

    for pat_id in id_list:
        pat_df = dataframe[dataframe['ID'] == pat_id].copy()
        if not pat_df.isnull().values.any() and len(pat_df.drop(['SepsisLabel', 'ID'], axis=1).values) == ts_pad_lenght:
            data_list.append(pat_df.drop(['SepsisLabel', 'ID'], axis=1).values)
            label_list.append(1 if pat_df.SepsisLabel.eq(1).any() else 0)

    # Convert to Pytorch dataset
    dataset = np.array(data_list)

    filtered_df = dataframe[dataframe['ID'].isin(id_list)]

    return dataset, data_list[0].shape[1], filtered_df


if __name__ == "__main__":

    print("Processing may take a few minutes")

    sepsis_path = '/home/adrien/Data/Sepsis/raw_data/training/'
    #sepsis_path = '/home/adrien/Data/Test_Sepsis/physionet.org/files/challenge-2019/1.0.0/training/'

    print(sepsis_path)

    dataframes = []

    for root, dirs, files in os.walk(sepsis_path, topdown=False):
        for name in files:
            if not os.path.join(root, name).endswith('.html'):
                dataframe = pd.read_csv(os.path.join(root, name), sep='|')
                dataframe = dataframe.drop(['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                                            'Alkalinephos', 'Calcium', 'Chloride', 'Magnesium', 'Phosphate',
                                            'Potassium',
                                            'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
                                            'Fibrinogen', 'Platelets', "O2Sat", "SBP", "DBP", "EtCO2", "Gender",
                                            "Unit1", "Unit2",
                                            'HospAdmTime'], axis=1)
                dataframe['ID'] = os.path.join(root, name).split('/')[-1].split('.')[0][1:]
                dataframes.append(dataframe)

    # Concatenate all DataFrames into one
    print("Concatenating all patient files into one dataframe...")
    final_dataframe = pd.concat(dataframes, ignore_index=True)

    print(len(final_dataframe))

    # Save unprocessed dataframe
    print("Saving the resulting dataframe...")
    path = "datasets/sepsis/processed_data"
    if not os.path.exists(path):
        os.makedirs(path)
    final_dataframe.to_csv(f"{path}/raw_sepsis_data.csv")

    # Create test and train dataframes
    dataset = "sepsis"
    train_ratio = 0.75
    no_sepsis_only = False
    ts_pad_len = 45
    batch_size = 32
    window_size = 6

    print("Generating train and test datasets...")
    train_data, test_data, num_features, train_df, test_df = get_train_test_loaders_encoding(
        data_path=f"{path}/raw_sepsis_data.csv",
        train_ratio=train_ratio,
        ts_pad_lenght=ts_pad_len,
        no_sepsis_only=no_sepsis_only
    )

    print("Saving datasets...")
    np.save(f'{path}/train_data.npy', train_data)
    np.save(f'{path}/test_data.npy', test_data)

    train_df.to_csv(f'{path}/clean_train_df.csv', index=False)
    test_df.to_csv(f'{path}/clean_test_df.csv', index=False)

    print("Data saved")
