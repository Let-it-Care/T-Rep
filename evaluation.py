import os 
import json
import torch
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from train_eval import train_and_test, Args
from utils import get_dataset_names


quick_uea_datasets = [
        'Libras', 'StandWalkJump', 'SelfRegulationSCP1', 'Handwriting',
       'JapaneseVowels', 'BasicMotions', 'ERing', 'MotorImagery',
       'FingerMovements', 'HandMovementDirection', 'SelfRegulationSCP2',
       'Heartbeat', 'EthanolConcentration', 'Cricket', 'PEMS-SF',
       'CharacterTrajectories', 'AtrialFibrillation', 'RacketSports',
       'UWaveGestureLibrary', 'ArticularyWordRecognition', 'Epilepsy',
       'DuckDuckGeese', 'NATOPS'
    ]


def save_evaluation_results(exp_path, num_runs, embedding_dims, dataset_gp, time_embedding, dfs):
    if not os.path.exists(f'{exp_path}'):
            os.makedirs(f'{exp_path}')
    
    filename = f'{exp_path}/d{embedding_dims}_{dataset_gp}'
    filename += f'_{time_embedding}' if time_embedding else ''

    if num_runs == 1:
        dfs[0].to_csv(f'{filename}.csv')
    else:
        dfs[0].to_csv(f'{filename}_mean.csv')
        dfs[1].to_csv(f'{filename}_std.csv')


########################## CLASSIFICATION ##########################

def eval_classification_ds(
        dataset_gp,
        dataset_name,
        embedding_dims,
        task_weights,
        epochs=40,
        time_embedding=None,
        num_runs=1,
        num_jobs=2,
        seed=1234,
    ):
    """
        Run classification task for a given dataset and
        latent space dimension and save results in a CSV.
    """
    print(f"Working on {dataset_name} classification ({num_runs} {'runs' if num_runs > 1 else 'run'}) ...")

    if not os.path.exists('cache'):
        os.makedirs('cache')

    def process_dataset(seed):
        args = Args(
            dataset=dataset_name,
            loader=dataset_gp,
            epochs=epochs,
            gpu=0,
            repr_dims=embedding_dims,
            time_embedding=time_embedding,
            task_weights=task_weights,
            seed=seed
        )
        eval_res = train_and_test(args, verbose=0)
        return np.array([eval_res['acc'], eval_res['auprc']]) 

    res = Parallel(n_jobs=num_jobs)(delayed(process_dataset)(seed) for _ in range(num_runs))
    return np.stack(res, axis=0)

def eval_classification_gp(
        dataset_gp,
        embedding_dims,
        exp_path,
        task_weights,
        time_embedding=None,
        epochs=40,
        num_runs=10,
        num_jobs=3,
        seed=1234,
        quick_eval=False,
    ):
    """
        Will train and test a model on each dataset
        of the UEA/UCR archive.

    Args:
        dataset_gp (_type_): UEA or UCR archive
        embedding_dims (_type_): Dimensionality of the time-series representation.
        exp_path (_type_): Experiment path to save results in.
        task_weights (_type_): Pretext task weights for the model.
        time_embedding (_type_, optional): Type of time-embedding to use. Defaults to None.
        epochs (int, optional): Number of training epochs. Defaults to 40.
        num_runs (int, optional): Number of model training and testing runs to do for each dataset. Defaults to 10.
        num_jobs (int, optional): Number of parallel jobs to use for faster execution. Defaults to 3.
        seed (int, optional): Random seed for reproducibility. Defaults to 1234.
        quick_eval (bool, optional): Run only a subset of (faster) classification datasets. Defaults to False.

    Returns:
        The classification results for the requested datasets.
    """

    if quick_eval and dataset_gp == "UEA":
        datasets = quick_uea_datasets
    else:
        datasets = get_dataset_names(dataset_gp)
    
    results = np.zeros((len(datasets), num_runs, 2))
    for i, ds in enumerate(datasets):
        print(f"{i + 1} / {len(datasets)} -", end=' ')
        t = time.time()
            
        ds_res = eval_classification_ds(
            dataset_gp=dataset_gp,
            dataset_name=ds,
            embedding_dims=embedding_dims,
            task_weights=task_weights,
            epochs=epochs,
            time_embedding=time_embedding,
            num_runs=num_runs,
            num_jobs=num_jobs,
            seed=seed,
        )

        print(f"{ds} res: {ds_res.mean(axis=0)}, {ds_res.shape} run time: {time.time() - t} seconds.")
        results[i] = ds_res
        np.save('cache/results_classif.npy', results)

    if num_runs == 1:
        res_df = pd.DataFrame(results.squeeze(), columns=['Accuracy', 'Avg. precision'], index=datasets)
        dfs = (res_df,)
    else:
        mean_df = pd.DataFrame(results.mean(axis=1), columns=['Accuracy', 'Avg. precision'], index=datasets)
        std_df = pd.DataFrame(results.std(axis=1), columns=['Accuracy', 'Avg. precision'], index=datasets)  
        dfs = (mean_df, std_df) 

    print(f"Saving results...", end=' ')
    save_evaluation_results(exp_path, num_runs, embedding_dims, dataset_gp, time_embedding, dfs)
    os.remove('cache/results_classif.npy')
    print("Done!")

    return res_df if num_runs == 1 else (mean_df, std_df)


########################## FORECASTING ##########################


def eval_ett_ds(
        dataset_name,
        embedding_dims,
        pred_horizons,
        num_jobs,
        task_weights,
        time_embedding=None,
        epochs=40,
        num_runs=10,
        seed=1234,
):
    def process_ett_ds(seed):
        args = Args(
            dataset=dataset_name,
            loader='forecast_csv',
            gpu=0,
            repr_dims=embedding_dims,
            epochs=epochs,
            task_weights=task_weights,
            time_embedding=time_embedding,
        )

        res = train_and_test(args, verbose=0)['ours']
        mse_norm = np.array([res[pred_h]['norm']['MSE'] for pred_h in pred_horizons])
        mae_norm = [res[pred_h]['norm']['MAE'] for pred_h in pred_horizons]

        mse_raw = np.array([res[pred_h]['raw']['MSE'] for pred_h in pred_horizons])
        mae_raw = [res[pred_h]['raw']['MAE'] for pred_h in pred_horizons]
        return np.stack((mse_norm, mae_norm, mse_raw, mae_raw), axis=1)

    res = Parallel(n_jobs=num_jobs, prefer='threads')(delayed(process_ett_ds)(seed) for _ in range(num_runs))
    return np.stack(res, axis=0)
    

def eval_all_ett(
        embedding_dims,
        exp_path,
        task_weights,
        time_embedding=None,
        epochs=40,
        num_runs=10,
        num_jobs=[10,10,1,10],
        seed=1234,
        save=True,
    ):
    """
        Train and test models on ETT datasets for forecasting tasks.

    Args:
        embedding_dims (_type_): Dimensionality of the time-series representations.
        exp_path (_type_): Path to save experience results in.
        task_weights (_type_): Pretext task weights for the model.
        time_embedding (_type_, optional): Type of time-embedding to use in the model. Defaults to None.
        epochs (int, optional): Number of epochs used in training. Defaults to 40.
        num_runs (int, optional): Number of train+test runs for each dataset, to obtain more stable results. Defaults to 10.
        num_jobs (list, optional): Number of parallel jobs, to speed up training. Defaults to [10,10,1,10].
        seed (int, optional): Random seed for reproducibility. Defaults to 1234.
        save (bool, optional): Save results or not. Defaults to True.

    Returns:
        Forecasting results (in a pandas dataframe format).
    """
    ett_datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']

    pred_horizons_h = [24, 48, 168, 336, 720]
    pred_horizons_m = [24, 48, 96, 288, 672]

    n_pred_horizons = len(pred_horizons_h)
    pred_scores = np.zeros((num_runs, len(ett_datasets) * n_pred_horizons, 4))

    for i, ds in enumerate(ett_datasets):
        print(f"{i + 1} / {len(ett_datasets)} - Working on {ds} forecasting ({num_runs} {'runs' if num_runs > 1 else 'run'}) ...")  
        pred_horizons = pred_horizons_h if 'h' in ds else pred_horizons_m

        res = eval_ett_ds(
            dataset_name=ds,
            embedding_dims=embedding_dims,
            task_weights=task_weights,
            pred_horizons=pred_horizons,
            time_embedding=time_embedding,
            epochs=epochs,
            num_runs=num_runs,
            num_jobs=num_jobs[i],
            seed=seed,
            
        )
        pred_scores[:, i * n_pred_horizons: (i + 1) * n_pred_horizons, :] = res
        np.save('cache/results_forecast.npy', pred_scores)

    pred_scores = pred_scores.swapaxes(0, 1)
    ds_index = list(np.repeat(ett_datasets, len(pred_horizons_h)))
    horizon_index = list(np.concatenate([
        pred_horizons_h * (len(ett_datasets) // 2),
        pred_horizons_m * (len(ett_datasets) // 2),
    ]))

    index = pd.MultiIndex.from_tuples(list(zip(ds_index, horizon_index)), names=['Dataset', 'Horizon'])
    cols = ["MSE_NORM", "MAE_NORM", "MSE_RAW", "MAE_RAW"]

    if num_runs > 1:
        mean_df = pd.DataFrame(pred_scores.mean(axis=1), columns=cols, index=index) 
        std_df = pd.DataFrame(pred_scores.std(axis=1), columns=cols, index=index) 
        dfs = (mean_df, std_df)
    else: 
        res_df = pd.DataFrame(pred_scores.squeeze(), columns=cols, index=index)
        dfs = (res_df,)

    print(f"D{embedding_dims} - Saving forecasting results...", end=' ')
    if save:
        save_evaluation_results(exp_path, num_runs, embedding_dims, "ETT", time_embedding, dfs)
    os.remove('cache/results_forecast.npy')
    print("Done!")    
    
    return dfs


########################## ANOMALY DETECTION ##########################


def eval_ad_yahoo(
        embedding_dims,
        exp_path,
        task_weights,
        epochs=40,
        num_runs=10,
        seed=None,
        time_embedding=None
    ):
    """
        Train model and run anomaly detection experiments on yahoo webscope S5
        datasets. 

    Args:
        embedding_dims (_type_): Dimensionality of the time-series representations.
        exp_path (_type_): Path to save experience results in.
        task_weights (_type_): Pretext task weights for the model.
        time_embedding (_type_, optional): Type of time-embedding to use in the model. Defaults to None.
        epochs (int, optional): Number of epochs used in training. Defaults to 40.
        num_runs (int, optional): Number of train+test runs for each dataset, to obtain more stable results. Defaults to 10.
        seed (int, optional): Random seed for reproducibility. Defaults to 1234.s to None.

    Returns:
        Anomaly detection results.
    """
    print(f"D{embedding_dims} - Working on anomaly detection...")
    
    def process_yahoo_ds(seed):
        args = Args(
                dataset='yahoo',
                loader='anomaly',
                batch_size=64,
                gpu=0,
                repr_dims=embedding_dims,
                task_weights=task_weights,
                epochs=epochs,
                time_embedding=time_embedding,
                seed=seed,
            )
        eval_res = train_and_test(args, verbose=0)
        return np.array([eval_res['f1'], eval_res['precision'], eval_res['recall']])

    res = np.zeros((num_runs, 3))
    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}...")
        res[i] = process_yahoo_ds(seed)
        np.save('cache/results_anomaly_detection.npy', res)

    if num_runs == 1:
        ad_yahoo_df = pd.DataFrame(res.squeeze(), columns=['f1', 'precision', 'recall'], index=['Yahoo'])
        dfs = (ad_yahoo_df,)
    else:
        mean_df = pd.DataFrame(res.mean(axis=0)[None, ...], columns=['f1', 'precision', 'recall'], index=['Yahoo']) 
        std_df = pd.DataFrame(res.std(axis=0)[None, ...], columns=['f1', 'precision', 'recall'], index=['Yahoo']) 
        dfs = (mean_df, std_df)

    print(f"D{embedding_dims} - Saving anomaly detection results...", end=' ')
    save_evaluation_results(exp_path, num_runs, embedding_dims, "Yahoo", time_embedding, dfs)
    os.remove('cache/results_anomaly_detection.npy')
    print("Done!")
    
    return dfs


def eval_one_dataset_classif(dataset_gp, dataset_name, epochs, embedding_dims, time_embedding, seed):
        args = Args(
            dataset=dataset_name,
            loader=dataset_gp,
            epochs=epochs,
            gpu=0,
            repr_dims=embedding_dims,
            time_embedding=time_embedding,
            seed=seed
        )
        eval_res = train_and_test(args, verbose=0)
        return np.array([eval_res['acc'], eval_res['auprc']]) 

def clear_gpu_memory():
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()

def save_exp_config(config, embedding_dims):
    exp_config = config.copy()
    exp_config['embedding_dims'] = embedding_dims

    nested_dict = exp_config['task_weights']
    del exp_config['task_weights']

    for key, value in nested_dict.items():
        new_key = key + "_weight"
        exp_config[new_key] = value

    exp_path = exp_config['exp_path']
    if not os.path.exists(f'{exp_path}'):
            os.makedirs(f'{exp_path}')
    with open(f"{exp_path}/config.json", "w") as fp:
        json.dump(exp_config, fp)