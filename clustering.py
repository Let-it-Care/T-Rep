import os 
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

import datautils
from utils import name_with_datetime, data_dropout, init_dl_program, get_dataset_names
from train_eval import train_trep, Args

interesting_datasets = [
    'SemgHandGenderCh2', 'CinCECGTorso', 'Phoneme', 'EOGVerticalSignal', 'WordSynonyms',
    'AllGestureWiimoteY', 'DodgerLoopWeekend', 'SemgHandMovementCh2', 'NonInvasiveFetalECGThorax1',
    'GestureMidAriD3', 'PigArtPressure', 'Lightning2', 'InlineSkate', 'SmoothSubspace', 
    'UWaveGestureLibraryAll', 'PLAID', 'SwedishLeaf', 'DistalPhalanxTW', 'PigCVP',
    'TwoLeadECG', 'Earthquakes', 'SonyAIBORobotSurface1', 'Haptics', 'Crop', 'FacesUCR', 
    'RefrigerationDevices', 'UWaveGestureLibraryY', 'NonInvasiveFetalECGThorax2', 
    'DodgerLoopGame', 'Chinatown', 'EOGHorizontalSignal', 'Fungi', 'ECGFiveDays',
    'AllGestureWiimoteZ', 'CricketX', 'OSULeaf', 'ElectricDevices', 'UWaveGestureLibraryZ', 
    'FaceAll', 'DistalPhalanxOutlineAgeGroup', 'MixedShapesSmallTrain', 'Mallat', 'CBF', 
    'MixedShapesRegularTrain', 'EthanolLevel', 'ProximalPhalanxTW', 'ChlorineConcentration', 
    'FreezerSmallTrain', 'FaceFour', 'ShapesAll', 'InsectWingbeatSound', 'MedicalImages', 
    'SemgHandSubjectCh2', 'MelbournePedestrian', 'CricketY', 'ShapeletSim', 'GesturePebbleZ1', 'ECG5000'
                        ]


def train_model(
        ds,
        ds_gp,
        epochs,
        task_weights,
        time_embedding,
        repr_dims,
        verbose,
    ):
    args = Args(
        dataset=ds,
        loader=ds_gp,
        epochs=epochs,
        gpu=0,
        repr_dims=repr_dims,
        time_embedding=time_embedding,
        task_weights=task_weights,
        seed=0,
    )

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    if ds_gp == "UEA":
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(ds)
    elif ds_gp == "UCR":
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(ds)

    if args.irregular > 0:
        train_data = data_dropout(train_data, args.irregular)
        test_data = data_dropout(test_data, args.irregular)

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    trep = train_trep(
        train_data,
        classif_args=None,
        forecast_args=None,
        ad_args=None,
        device=device,
        config=config,
        run_dir=run_dir,
        verbose=verbose,
        args=args
    )
    return trep, train_data, train_labels, test_data, test_labels


def run_clustering_exp(
    ds_gp,
    repr_dims,
    time_embedding,
    trep_task_weights,
    ts2vec_task_weights,
    results_folder="clustering_exp"
):
    uea_datasets = get_dataset_names(ds_gp)
    for i, ds in enumerate(uea_datasets):
        print(f"({i+1}/{len(uea_datasets)}) - {ds_gp} - CLUSTERING {ds}...")

        # Train the self-supervised models
        trep, train_data, train_labels, test_data, test_labels = train_model(
            ds=ds,
            ds_gp=ds_gp,
            epochs=epochs,
            task_weights=trep_task_weights,
            time_embedding=time_embedding,
            repr_dims=repr_dims,
            verbose=1
        )

        ts2vec, train_data, train_labels, test_data, test_labels = train_model(
            ds=ds,
            ds_gp=ds_gp,
            epochs=epochs,
            task_weights=ts2vec_task_weights,
            time_embedding=None,
            repr_dims=repr_dims,
            verbose=1
        )

        # Get the time-series representation
        trep_representations = trep.encode(test_data, encoding_window='full_series', batch_size=64)
        ts2vec_representations = ts2vec.encode(test_data, encoding_window='full_series', batch_size=64)

        classes = np.unique(train_labels)
        n_classes = len(classes)
        clusters = test_labels

        # Apply UMAP for dimensionality reduction
        umap_params = {
            'n_neighbors': 10,
            'min_dist': 0.1,
            'n_components': 2,
        }
        trep_umap = UMAP(n_neighbors=umap_params['n_neighbors'], min_dist=umap_params['min_dist'], n_components=umap_params['n_components'], random_state=2)
        trep_embed = trep_umap.fit_transform(trep_representations)

        ts2vec_umap = UMAP(n_neighbors=umap_params['n_neighbors'], min_dist=umap_params['min_dist'], n_components=umap_params['n_components'], random_state=2)
        ts2vec_embed = ts2vec_umap.fit_transform(ts2vec_representations)

        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, n_classes))

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        for c in classes:
            trep_idcs = np.where(clusters == c)[0]
            ax[0].scatter(trep_embed[trep_idcs, 0], trep_embed[trep_idcs, 1], c=[colors[c]] * len(trep_idcs), s=20, label=c)
            ax[0].set_title(f"T-Rep - {ds}")

            ts2vec_idcs = np.where(clusters == c)[0]
            ax[1].scatter(ts2vec_embed[ts2vec_idcs, 0], ts2vec_embed[ts2vec_idcs, 1], c=[colors[c]] * len(ts2vec_idcs), s=20)
            ax[1].set_title(f"TS2Vec - {ds}")

        fig.legend(loc='lower center', mode='expand', bbox_to_anchor=(0.48, 0.00, 0.05, 0.02))
        plt.savefig(f"{results_folder}/{ds_gp}_{ds}.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    ds_gp = 'UCR'
    epochs = 40
    trep_task_weights = {
            'instance_contrast': 0.25,
            'temporal_contrast': 0.25,
            'tembed_jsd_pred': 0.25,
            'tembed_cond_pred': 0.25, 
        }
    ts2vec_task_weights = {
            'instance_contrast': 0.5,
            'temporal_contrast': 0.5,
            'tembed_jsd_pred': 0.0,
            'tembed_cond_pred': 0.0, 
        }
    time_embedding = 't2v_sin'
    repr_dims = 128
    
    run_clustering_exp(
        epochs=40,
        ds_gp="UCR",
        repr_dims=repr_dims,
        time_embedding=time_embedding,
        trep_task_weights=trep_task_weights,
        ts2vec_task_weights=ts2vec_task_weights,
        results_folder="clustering_exp"
    )

    run_clustering_exp(
        epochs=40,
        ds_gp="UEA",
        repr_dims=repr_dims,
        time_embedding=time_embedding,
        trep_task_weights=trep_task_weights,
        ts2vec_task_weights=ts2vec_task_weights,
        results_folder="clustering_exp"
    )