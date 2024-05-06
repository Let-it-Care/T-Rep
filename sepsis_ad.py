import tasks
import torch
import numpy as np
import pandas as pd

from train_eval import train_trep, Args

SEPSIS_DATA_PATH = 'datasets/sepsis'
N_TEST_PATIENTS = 10084

def train_trep_sepsis(
        task_weights,
        time_embedding,
        repr_dims,
        epochs,
        model_name='',
        hidden_dims=64,
        seed=None
    ):

    # T-Rep config
    trep_args = Args(
        dataset="",
        loader="",
        epochs=epochs,
        gpu=0,
        repr_dims=repr_dims,
        time_embedding=time_embedding,
        task_weights=task_weights,
        batch_size=32,
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    config = dict(
        batch_size=trep_args.batch_size,
        lr=trep_args.lr,
        output_dims=trep_args.repr_dims,
        max_train_length=trep_args.max_train_length,
        hidden_dims=hidden_dims,
    )
    model = 'trep' if time_embedding is not None else 'ts2vec'
    run_dir = f"{model}{model_name}_l{repr_dims}_e{epochs}_{time_embedding}"


    # Load data
    print(f"Loading anomaly detection data...")
    train_data = np.load(f'{SEPSIS_DATA_PATH}/processed_data/train_data.npy')
    test_data = np.load(f'{SEPSIS_DATA_PATH}/processed_data/test_data.npy')

    # Train and save model
    trep = train_trep(
        train_data=train_data,
        device=device,
        config=config,
        verbose=1,
        run_dir=run_dir,
        args=trep_args,
    )

    print("Training of embedding model done.")
    print(f"Saved at {run_dir}")
    
    return trep


def get_patient_windows(df, window_size):
    windows = []
    for i in range(window_size, len(df) + 1):
        windows.append(df.iloc[i - window_size:i])
    return windows


def get_window_xy(window_df):
    return window_df.drop(['SepsisLabel', 'ID'], axis=1).values, window_df['SepsisLabel'].max()


def get_patient_ds(patient_id, patient_dfs, window_size):
    df = patient_dfs[patient_id]
    if window_size == 0:
        return df.drop(['SepsisLabel', 'ID'], axis=1).values, df['SepsisLabel'].values
    window_dfs = get_patient_windows(df, window_size)
    F = window_dfs[0].shape[1] - 2
    X = np.zeros((len(window_dfs), 6, F))
    y = np.zeros((len(window_dfs), 1))
    for i, w in enumerate(window_dfs):
        X[i], y[i] = get_window_xy(w) 
    return X, y


def get_sepsis_ad_df(df, window_size):
    ids = df['ID'].unique()
    patient_dfs = {pid: df[df['ID'] == pid] for pid in ids}
    patient_dss = [get_patient_ds(p, patient_dfs, window_size) for p in ids]
    Xs, ys = zip(*patient_dss)
    if window_size == 0:
        X = np.stack(Xs, axis=0)
        y = np.stack(ys, axis=0)
    else:
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
    return X, y


def get_pointwise_labels(df):
    ids = df['ID'].unique()
    patient_dfs = {pid: df[df['ID'] == pid] for pid in ids} 
    patient_labels = np.zeros((len(ids), 45))
    for i, (pid, patient_df) in enumerate(patient_dfs.items()):
        patient_labels[i] = patient_df['SepsisLabel'].values
    return patient_labels


def run_sepsis_exp(
    repr_dims,
    epochs,
    model_type='trep',
    window_size=6,
    seed=0,
):
    # Train TRep on Sepsis data
    print(f"Creating and training {model_type} model...")
    if model_type != 'raw_data':
        hidden_dims = np.array([128, 128, 128, 128, 64, 64, 64, 64, 32, 32])
        task_weights = {
            'instance_contrast': 0.25,
            'temporal_contrast': 0.25,
            'tembed_jsd_pred': 0.25,
            'tembed_cond_pred': 0.25, 
        }
        time_embedding = 't2v_sin'

        model = train_trep_sepsis(
            task_weights,
            time_embedding,
            repr_dims,
            epochs,
            hidden_dims=hidden_dims,
            seed=seed
        )

    # Build datasets for segment-based anomaly detection
    print(f"Loading anomaly detection data...")
    train_df = pd.read_csv(f"{SEPSIS_DATA_PATH}/processed_data/clean_train_df.csv", index_col=0)
    test_df = pd.read_csv(f"{SEPSIS_DATA_PATH}/processed_data/clean_test_df.csv", index_col=0)
    train_X, train_y = get_sepsis_ad_df(train_df, window_size=window_size)
    test_X, test_y = get_sepsis_ad_df(test_df, window_size=window_size)
    print(f"Train X: {train_X.shape}, train y: {train_y.shape}")
    print(f"Test X: {test_X.shape}, test y: {test_y.shape}")
    test_labels = np.zeros((N_TEST_PATIENTS, 45))
    test_labels[:, 5:] = test_y.reshape(-1, 40)

    # Train and test SVM for segment-based anomaly detection
    print(f"Training and testing anomaly detection model...")
    test_preds, test_accs = tasks.eval_anomaly_detection_sepsis(
        model=model if model_type == 'trep' else None,
        train_data=train_X,
        train_labels=train_y.squeeze(),
        test_data=test_X,
        test_labels=test_labels,
        eval_protocol='svm',
        window_size=window_size,
        raw_data=(model_type == 'raw_data')
    )

    return test_accs


if __name__ == "__main__":

    test_accs_trep = run_sepsis_exp(
        repr_dims=32,
        epochs=40,
        model_type='trep',
        window_size=6,
        seed=0
    )
