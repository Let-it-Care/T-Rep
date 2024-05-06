# T-Rep

This repository contains the official implementation for the paper ["T-Rep: Representation Learning for Time-Series Using Time-Embeddings"](https://arxiv.org/abs/2310.04486).

It was built on top of the [TS2Vec repository](https://github.com/yuezhihan/ts2vec), which provided a very good start point for both model development and benchmarking. A big thanks to the authors!

## Requirements

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```
The repository is not yet compatible with Pytorch 2.0. It includes specific (not always the latest) versions of packages, so we recommend having a dedicated virtual environment for this repo.

## Usage

### Command line

To train and evaluate T-Rep on one of the supported datasets (see below), run the following command:

```train & evaluate
python train.py <dataset_name> <run_name> --loader <loader> --repr-dims <repr_dims> --eval
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader | The data loader used to load the dataset. This can be set to `UCR`, `UEA`, `forecast_csv`, `forecast_csv_univar`, `anomaly`, or `anomaly_coldstart` |
| repr_dims | The representation dimensions (defaults to 320) |
| eval | Whether to perform evaluation after training |

(For descriptions of more arguments, run `python train.py -h`.)

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/DatasetName__RunName_DateTime/`. 

### Code

A detailed tutorial of how to use T-Rep is provided in ``full_tutorial.ipynb``, but we showcase below the simple sklearn-like interface used by T-Rep.

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from trep import TRep
import datautils

# Load the ECG200 dataset from UCR archive
train_data, train_labels, test_data, test_labels = datautils.load_UCR('ECG200')
# (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)

# Instantiate and train T-Rep
trep = TRep(
    input_dims=1,
    device=0,
    time_embedding='t2v_sin',
    output_dims=128
)
loss_log = trep.fit(train_data, n_epochs=80, verbose=1)

# Compute timestamp-level representations for test set
train_repr = trep.encode(train_data)  # n_instances x n_timestamps x output_dims
test_repr = trep.encode(test_data)  # n_instances x n_timestamps x output_dims


# Classify the learned representations using an SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(train_repr, train_labels)
y_pred = svm_classifier.predict(test_repr)
accuracy = accuracy_score(test_labels, y_pred)
```

This is all you need to know to use T-Rep. The produced `np.ndarray` of representations can then be used as inputs for any task ranging from classification, clustering, forecasting, to anomaly detection etc.

## Reproduction of Results


### Data

The datasets used in the paper to evaluate the model can be downloaded from:

* [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into `datasets/UCR/` so that each data file can be located by `datasets/UCR/<dataset_name>/<dataset_name>_*.csv`.
* [30 UEA datasets](http://www.timeseriesclassification.com) should be put into `datasets/UEA/` so that each data file can be located by `datasets/UEA/<dataset_name>/<dataset_name>_*.arff`.
* [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/ETTh1.csv`, `datasets/ETTh2.csv` and `datasets/ETTm1.csv`.
* [Yahoo dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70) should be preprocessed using `datasets/preprocess_yahoo.py` and placed at `datasets/yahoo.pkl`.
* [Sepsis dataset](https://physionet.org/content/challenge-2019/1.0.0/training/#files-panel) files should be placed under `datasets/Sepsis` and preprocessed using `datasets/preprocess_sepsis.py`.

### Runnning Experiments

All functions necessary to reproduce experiments and results shown in the T-Rep paper are provided in this repository. For reproduction and experiment details, please refer directly to the paper (Appendix A.2).

- **Classification, Forecasting, Anomaly Detection**: To reproduce experiments for these tasks, you can use functions in `evaluation.py`.
- **Clustering**: Clustering experiments can be reproduced using functions in the `clustering.py` file. Example parameterisation and function calls are provided at the bottom of the file, in the `__main__` function.
- **Sepsis**: The code to reproduce Sepsis anomaly detection results can be found in the `sepsis_ad.py` file. An example function call is given in the `__main__` function.

## Citations

If this work has proven useful or you are using this repository for your project, please cite using

```bibtex
@inproceedings{
    fraikin2024trep,
    title={T-Rep: Representation Learning for Time Series using Time-Embeddings},
    author={Archibald Felix Fraikin and Adrien Bennetot and Stephanie Allassonniere},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=3y2TfP966N}
}
```


