import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .dilated_conv import DilatedConvEncoder
from .time_embeddings import (
    LearnablePositionalEncodingSmall,
    LearnablePositionalEncodingBig,
    LearnablePositionalEncodingHybrid,
    T2vCos,
    T2vSin,
    GaussianPositionalEncoding
)

time_embeddings = {
    "t2v_sin": T2vSin,
    "t2v_cos": T2vCos,
    "fully_learnable_big": LearnablePositionalEncodingBig,
    "fully_learnable_small": LearnablePositionalEncodingSmall,
    "gaussian": GaussianPositionalEncoding,
    "hybrid": LearnablePositionalEncodingHybrid,
}

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TSEncoder(nn.Module):
    def __init__(
            self,
            input_dims,
            output_dims,
            hidden_dims=64,
            depth=10,
            mask_mode='binomial',
            time_embedding=None,
            time_embedding_dim=64
        ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        fc_dim = hidden_dims if isinstance(hidden_dims, int) else hidden_dims[0]
        self.input_fc = nn.Linear(input_dims, fc_dim)

        if time_embedding:
            self.time_embedding = time_embeddings[time_embedding](in_features=1, out_features=time_embedding_dim)
            self.time_embedding_dim = time_embedding_dim
        else:
            self.time_embedding = None

        feature_extractor_dims = hidden_dims + time_embedding_dim if time_embedding else hidden_dims
        if isinstance(hidden_dims, np.ndarray):
            assert depth == len(hidden_dims)
            self.feature_extractor = DilatedConvEncoder(
                feature_extractor_dims[0],
                list(feature_extractor_dims) + [output_dims],
                kernel_size=3
            )
        else:
            self.feature_extractor = DilatedConvEncoder(
                feature_extractor_dims,
                [feature_extractor_dims] * depth + [output_dims],
                kernel_size=3
            )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, time_vec, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0

        # Add positional encoding
        t_embed = None
        if self.time_embedding:

            # Handle padding with nans by setting the time embeddings to 0.
            mask = (time_vec != torch.full_like(time_vec, -1).to(x.device)).type(torch.float16)
            mask = mask.repeat(1, 1, self.time_embedding_dim)
            t_embed = self.time_embedding(time_vec)
            t_embed = mask * t_embed

            x = torch.cat([x, t_embed], dim=-1)
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Ch x T
        x = x.transpose(1, 2)  # B x T x Ch

        return x, t_embed
        