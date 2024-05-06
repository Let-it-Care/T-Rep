import os
import time
import datetime
from dataclasses import dataclass

import tasks
import datautils
from trep import TRep
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout


@dataclass
class Args:
    dataset: str
    loader: str
    gpu: int
    time_embedding: str
    repr_dims: int
    epochs: int
    task_weights: dict
    run_name: str = ""
    batch_size: int = 16 
    lr: float = 0.001
    max_train_length = 800
    iters: int = None
    save_every = None
    seed: int = 1234
    max_threads = None
    eval: bool = True
    irregular = 0


def save_checkpoint_callback(
    run_dir,
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback


def train_trep(train_data, device, config, run_dir, verbose, args):
    """
        Creates an instance of the model and launches training with
        the given parameters.
    
    Args:
        train_data (np.ndarray): Data used to train the model.
        device (_type_): Device to train the model on (gpu, cpu).
        config (dict): Configuration settings for initialising the model.
        run_dir (str): Directory in which to save the trained mdoel.
        verbose (int): Verbosity level (0 to 2).
        args (dataclass): Additional arguments used to configure model or training settings.

    Returns:
        The trained model.
    """
    t = time.time()
    
    model = TRep(
        input_dims=train_data.shape[-1],
        device=device,
        time_embedding=args.time_embedding,
        task_weights=args.task_weights,
        **config
    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=verbose,
    )

    run_dir = f"training/{run_dir}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    model.save(f'{run_dir}/model.pt')

    t = time.time() - t
    if verbose >= 1:
        print(f"Training time: {datetime.timedelta(seconds=t)}")

    return model

def train_and_test(args, verbose=1, save_eval_res=False):
    """
        This function is the entrypoint of the repo. It loads the desired
        dataset, creates and trains a model on the dataset, and then evaluates it.
        It does so for classification, forecasting, and anomaly detection tasks.

    Args:
        args: Arguments to setup dataset, model, training and evaluation settings.
                These can either come from command line arguments or the Args dataclass.
        verbose (int, optional): Verbosity level (0-2).

    Returns:
        The evaluation results on the requested task.
    """

    if verbose >= 1:
        print("Dataset:", args.dataset)
    if verbose >= 2:
        print("Arguments:", str(args))

    ##################### CONIGURATION ####################
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    print(f"Device used: {device}")
    
    if verbose >= 2:
        print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
        
        
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
        
    if verbose >= 2:
        print('Done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(run_dir, args.save_every, unit)

    ####################### TRAIN MODEL ######################

    model = train_trep(train_data, device, config, run_dir, verbose, args)

    ####################### MODEL EVALUATION ####################

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(
                model,
                train_data,
                train_labels,
                test_data,
                test_labels,
                encoding_protocol='full_series',
                eval_protocol='svm'
            )
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(
                model,
                data,
                train_slice,
                valid_slice,
                test_slice,
                scaler,
                pred_lens,
                args.time_embedding is not None
            )
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(
                model,
                all_train_data,
                all_train_labels,
                all_train_timestamps,
                all_test_data,
                all_test_labels,
                all_test_timestamps,
                delay, verbose=True
            )
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(
                model,
                all_train_data,
                all_train_labels,
                all_train_timestamps,
                all_test_data,
                all_test_labels,
                all_test_timestamps,
                delay
            )
        else:
            assert False

        if save_eval_res:
            pkl_save(f'{run_dir}/out.pkl', out)
            pkl_save(f'{run_dir}/eval_res.pkl', eval_res)

        if verbose >= 1:
            print(f'Evaluation result: {eval_res}\n')

    if verbose >= 2:
        print("Finished.")
    return eval_res
