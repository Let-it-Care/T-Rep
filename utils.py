import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime
from sklearn.utils import resample

def get_dataset_names(dataset_gp):
    raw_list = [x[0].split('/')[2] for x in os.walk(f'datasets/{dataset_gp}/')]
    raw_list.remove('')
    blacklist = ['Images', 'Descriptions', 'Missing_value_and_variable_length_datasets_adjusted']
    whitelist = lambda x: np.all([x not in b for b in blacklist])
    filtered_list = [ds for ds in raw_list if whitelist(ds)]
    return filtered_list

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan).to(arr.device), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan).to(arr.device)), dim=dim)
    return arr

def torch_pad_with(arr, pad_val, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, pad_val).to(arr.device), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, pad_val).to(arr.device)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
         arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    """
        Takes num_elements starting at indx for each batch element.
    """
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t if torch.cuda.is_available() else "cpu")
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return devices if len(devices) > 1 else devices[0]


def ot_batch_dist(X, Y, squared):
    """
        Compute distance matrix for Wasserstein distance computation
        from matrices of shape (B, T, C), computing distances along
        the final axis (C).
    """
    a2 = torch.einsum('btij,btij->bti', X, X)
    b2 = torch.einsum('btij,btij->bti', Y, Y)

    c = -2 * torch.einsum('btij, btkj -> btik', X, Y)

    c += a2[..., None]
    c += b2[:, :, None, :]

    c = torch.maximum(c, torch.zeros_like(c).to(X.device))
    if not squared:
        c = torch.sqrt(c)

    return c # B x T x C x C


def torch_kl(a, b):
    zero_constant = 0.00000001
    a_nozero = torch.where(a == 0, a + zero_constant, a)
    b_nozero = torch.where(b == 0, b + zero_constant, b)
    return a_nozero * (torch.log(a_nozero) - torch.log(b_nozero))


def find_closest_train_segment(train_data, test_data, step=1, squared_dist=True):
    """
        Find the starting indices of segments in the training dataset with the smallest squared
        Euclidean distances to corresponding segments in the test dataset.

        Parameters:
        - train_data: 3D NumPy array (B: batches, T: timesteps, _: features). Must have the same batch size as the test set.
        - test_data: 3D NumPy array (B: batches, t: timesteps, _: features).  Must have the same batch size as the train set.
        - step: Optional parameter specifying the step size for iterating over the training data. Default is 1.
        - squared_dist: Optional parameter specifying whether to compute squared distances. Default is True.

        Returns:
        - time_indices: 3D NumPy array containing the starting time indices for segments in the
                        training dataset with the smallest squared Euclidean distances to the test dataset.
                        Shape: (B, 1, t), where B is the number of batches and t is the number of timesteps in the test dataset.
    """
    B, T, _ = train_data.shape
    t = test_data.shape[1]

    if t == T:
        return np.tile(np.arange(T), (B, 1))[..., None]

    dist = np.zeros((B, (T - t) // step))
    for i in range(0, T - t, step):
        if squared_dist:
            dist[:, i] = np.abs(np.mean(np.sum((train_data[:, i: i + t, :] - test_data[:, :, :]) ** 2, axis=1), axis=1))
        else:
            dist[:, i] = np.abs(np.mean(np.sum(train_data[:, i: i + t, :] - test_data[:, :, :], axis=1), axis=1))

        min_dist = np.argmin(dist, axis=1)
        seq_starts = (step * min_dist)[..., None]
    
    # Get time indices for each segment
    time_indices = (np.arange(t) + seq_starts).astype(np.int32)[..., None]
    return time_indices


def upsample_minority_class(train_repr, train_labels, upsample_ratio=1.0):
    pos_idcs = np.where(train_labels == 1.0)[0]
    neg_idcs = np.where(train_labels == 0.0)[0]
    pos_train_repr = train_repr[pos_idcs]
    bootstrapped_train_repr = resample(pos_train_repr, 
                                replace=True,
                                n_samples=int(len(neg_idcs) * upsample_ratio),  
                                random_state=123)
    labels = np.concatenate(
        [np.zeros(len(neg_idcs)), np.ones(len(bootstrapped_train_repr))]
    )
    reprs = np.concatenate(
        [train_repr[neg_idcs], bootstrapped_train_repr],
        axis=0
    )

    shuffled_idcs = np.random.RandomState(seed=42).permutation(len(labels))
    reprs = reprs[shuffled_idcs]
    labels = labels[shuffled_idcs]
    return reprs, labels