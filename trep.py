import time
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from models import TSEncoder
from models.task_heads import TembedDivPredHead, TembedCondPredHead, TembedKLPredHeadLinear
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan, torch_pad_with

class TRep:
    '''The TRep model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=128,
        hidden_dims=np.array([64, 128, 256, 256, 256, 256, 128, 128, 128, 128]),
        time_embedding=None,
        time_embedding_dim=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        task_weights=None,
        max_train_length=None,
        temporal_unit=0,
    ):
        ''' Initialize a TRep model.
        
        Args:
            input_dims (int): The input dimension (=number of channels of time series). For a univariate time series, this should be set to 1.
            output_dims (int): The representation (=latent space) dimension.
            hidden_dims : The hidden dimension of the encoder. Can be an int (all layers will have same width), or an ndarray specifying width of each layer.
            time_embedding (str): The type of the time-embedding to be used.
            time_embedding_dim (int): The number of output dimensions of the time-embedding module.
            depth (int): The number of hidden residual blocks in the encoder.
            device (str): The device ('cpu', 'cuda'...) used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            Task weights (dict): The weights to assign to each pretext task during training.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.time_embedding = time_embedding
        self._net = TSEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            time_embedding=self.time_embedding,
            time_embedding_dim=time_embedding_dim,
        ).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        if task_weights is not None:
            self.task_weights = task_weights
        else:
            self.task_weights = {
               'instance_contrast': 0.25,
                'temporal_contrast': 0.25,
                'tembed_jsd_pred': 0.25,
                'tembed_cond_pred': 0.25, 
            }
        
        assert sum(self.task_weights.values()) == 1.0

        self.tembed_jsd_task_head = TembedDivPredHead(
            in_features=output_dims,
            out_features=1,
            hidden_features=128
        ).to(self.device)

        self.tembed_pred_task_head = TembedCondPredHead(
            in_features=output_dims + time_embedding_dim,
            hidden_features=[64,128],
            out_features=output_dims,
        ).to(self.device)
        
        self.n_epochs = 0
        self.n_iters = 0
    
    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=0):
        ''' Training the TRep model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        
        # Split data into windows, pad windows with nans to have equal lengths
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        # What timesteps have no modalities present for at least one batch element
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        # Eliminate empty series        
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        if verbose:
            print(f"Training data shape: {train_data.shape}")
        
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        train_start = time.time()
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                # Batch is a 1 element list
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                # Time vector to pass to model
                ts_l = x.size(1)
                time_vec = torch.arange(
                    ts_l,
                    dtype=torch.float32,
                ).repeat(x.shape[0], 1)[..., None]
                time_vec = time_vec.to(self.device)
                
                # Choose overlapping windows for timestamp masking (batch-wise)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                # Crop offset gives different windows for the different elements in the batch
                # Bounds how much can deviate from a1/a2/b1/b2 while being in [0, T].
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                
                x1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
                time1 = take_per_row(time_vec, crop_offset + crop_eleft, crop_right - crop_eleft)
                out1, tau1 = self._net(x1, time1)
                out1 = out1[:, -crop_l:]
                if tau1 is not None:
                    tau1 = tau1[:, -crop_l:]
                
                x2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
                time2 = take_per_row(time_vec, crop_offset + crop_left, crop_eright - crop_left) 
                out2, tau2 = self._net(x2, time2)
                out2 = out2[:, :crop_l]
                if tau2 is not None:
                    tau2 = tau2[:, :crop_l]
                
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    tau1,
                    tau2,
                    tembed_pred_task_head=self.tembed_pred_task_head,
                    tembed_jsd_task_head=self.tembed_jsd_task_head,
                    temporal_unit=self.temporal_unit,
                    weights=self.task_weights
                )
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose >= 2:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

        return loss_log
    
    def _eval_with_pooling(
            self,
            x,
            time_vec,
            mask=None,
            slicing=None,
            encoding_window=None,
        ):
        out, time_embeddings = self.net(x.to(self.device, non_blocking=True), time_vec.to(self.device, non_blocking=True), mask)
        
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
                time_embeddings = time_embeddings[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            if time_embeddings is not None:
                time_embeddings = F.max_pool1d(
                    time_embeddings.transpose(1, 2),
                    kernel_size = time_embeddings.size(1),
                ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = encoding_window // 2,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if time_embeddings is not None:
                time_embeddings = F.max_pool1d(
                    time_embeddings.transpose(1, 2),
                    kernel_size = encoding_window,
                    stride = 1,
                    padding = encoding_window // 2
                ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
                if time_embeddings is not None:
                    time_embeddings = time_embeddings[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
                if time_embeddings is not None:
                    time_embeddings = time_embeddings[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            tembeds = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if time_embeddings is not None:
                    t_tembed = F.max_pool1d(
                        time_embeddings.transpose(1, 2),
                        kernel_size = (1 << (p + 1)) + 1,
                        stride = 1,
                        padding = 1 << p
                    ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                    if time_embeddings is not None:
                        t_tembed = t_tembed[:, slicing]
                reprs.append(t_out)
                if time_embeddings is not None:
                    tembeds.append(t_tembed)
                p += 1
            out = torch.cat(reprs, dim=-1)
            if time_embeddings is not None:
                time_embeddings = torch.cat(tembeds, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
                if time_embeddings is not None:
                    time_embeddings = time_embeddings[:, slicing]
            
        return out.cpu(), time_embeddings.cpu() if time_embeddings is not None else None

    
    def encode(
            self,
            data,
            time_indices=None,
            mask=None,
            encoding_window=None,
            causal=False,
            sliding_length=None,
            sliding_padding=0,
            batch_size=None,
            return_time_embeddings=False,
        ):
        ''' Compute representations using the model.
        
        Args:
            data (np.ndarray): This should have a shape of (n_instances, n_timestamps, n_features). All missing data should be set to NaN.
            time_indices (np.ndarray): Timestep indices to be fed to the time-embedding module. The 'find_closest_train_segment' from tasks.forecasting.py can be used to find timesteps at which the test set most resembles the train set.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'. It is used for anomaly detection, otherwise left to 'None'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation undergoes max pooling with kernel size determined by this param. It can be set to 'full_series' (Collapsing the time dimension of the representation to 1), 'multiscale' (combining representations at different time-scales) or an integer specifying the pooling kernel size. Leave to 'None' and no max pooling will be applied to the time dimension: it will be the same as the raw data's.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp. This is done using causal convolutions.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            return_time_embeddings (bool): Whether to only return the encoded time-series representations, or also the associated time-embedding vectors.
            
        Returns:
            repr, time_embeddings: 'repr' designates the representations for the input time series. The representation's associated time-embeddings are also returned if 'return_time_embeddings' is set to True.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        if time_indices is not None:
            dataset = TensorDataset(torch.from_numpy(data).to(torch.float), torch.from_numpy(time_indices).to(torch.float))
        else:
            dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
 
        with torch.no_grad():
            output = []
            output_time_embeddings = []
            for batch in loader:

                if time_indices is not None:
                    x, full_time_vec = batch
                else:
                    x = batch[0]
                    full_time_vec = torch.arange(
                        ts_l,
                        dtype=torch.float32,
                    ).repeat(x.shape[0], 1)[..., None]

                if sliding_length is not None:
                    reprs = []
                    tembeds = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                        time_vec_buffer = []
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        time_vec_sliding = torch_pad_with(
                            full_time_vec[:, max(l, 0) : min(r, ts_l)],
                            pad_val=-1,
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )

                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out, time_embeddings = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    torch.cat(time_vec_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window,
                                )
                                reprs += torch.split(out, n_samples)
                                if return_time_embeddings:
                                    tembeds += torch.split(time_embeddings, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                                time_vec_buffer = []
                            calc_buffer.append(x_sliding)
                            time_vec_buffer.append(time_vec_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out, time_embeddings = self._eval_with_pooling(
                                x_sliding,
                                time_vec_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window,
                            )
                            reprs.append(out)
                            if return_time_embeddings:
                                tembeds.append(time_embeddings)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out, time_embeddings = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                torch.cat(time_vec_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window,
                            )
                            reprs += torch.split(out, n_samples)
                            if return_time_embeddings:
                                tembeds += torch.split(time_embeddings, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if return_time_embeddings:
                        time_embeddings = torch.cat(tembeds, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                        if return_time_embeddings:
                            time_embeddings = F.max_pool1d(
                                time_embeddings.transpose(1, 2).contiguous(),
                                kernel_size = time_embeddings.size(1),
                            ).squeeze(1)
                else:
                    out, time_embeddings = self._eval_with_pooling(
                        x,
                        full_time_vec,
                        mask,
                        encoding_window=encoding_window,
                    )
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        if return_time_embeddings:
                            time_embeddings = time_embeddings.squeeze(1)
                        
                output.append(out)
                if return_time_embeddings:
                    output_time_embeddings.append(time_embeddings)
                
            output = torch.cat(output, dim=0)
            if return_time_embeddings:
                output_time_embeddings = torch.cat(output_time_embeddings, dim=0)
            
        self.net.train(org_training)

        if return_time_embeddings:
            return output.numpy(), output_time_embeddings
        else:
            return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    
