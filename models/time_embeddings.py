import torch
import math
from torch import nn

# Time2Vec implementation from https://github.com/ojus1/Time2Vec-PyTorch

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
         v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class T2vSin(nn.Module):
    def __init__(self, in_features, out_features, normalise=True):
        super(T2vSin, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin
        self.sigmoid = nn.Sigmoid()
        self.normalise = normalise

    def forward(self, tau):
        out = t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
        out = self.sigmoid(out)
        if self.normalise:
            return out / torch.sum(out, dim=-1)[..., None]


class T2vCos(nn.Module):
    def __init__(self, in_features, out_features):
        super(T2vCos, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tau):
        out = t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
        out = self.sigmoid(out)
        return out / torch.sum(out, dim=-1)[..., None]


class LearnablePositionalEncodingHybrid(nn.Module):

    def __init__(self, in_features, out_features, hidden_features=32, dropout=0.1):
        super(LearnablePositionalEncodingHybrid, self).__init__()

        self.t2v_sin = T2vSin(
            in_features=in_features,
            out_features=out_features // 2,
            normalise=False
        )
        self.t2v_fc = LearnablePositionalEncodingBig(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features // 2,
            dropout=dropout,
            normalise=False
        )

    def forward(self, x):
        out_t2v_sin = self.t2v_sin(x)
        out_t2v_fc = self.t2v_fc(x)
        out = torch.cat((out_t2v_sin, out_t2v_fc), dim=-1)
        return out / torch.sum(out, dim=-1)[..., None]


class LearnablePositionalEncodingSmall(nn.Module):

    def __init__(self, in_features, out_features, hidden_features=32, dropout=0.1):
        super(LearnablePositionalEncodingSmall, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, t):
        out = self.sigmoid(self.fc(t))
        return out / torch.sum(out, dim=-1)[..., None]


class LearnablePositionalEncodingBig(nn.Module):

    def __init__(self, in_features, out_features, hidden_features=32, dropout=0.1, normalise=True):
        super(LearnablePositionalEncodingBig, self).__init__()
        self.normalise = normalise
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            self.relu,
            nn.Linear(hidden_features, out_features),
            self.sigmoid,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.dropout(self.fc(x))
        if self.normalise:
            return out / torch.sum(out, dim=-1)[..., None]


class GaussianPositionalEncoding(nn.Module):

    def __init__(self, in_features, out_features):
        super(GaussianPositionalEncoding, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigmoid = nn.Sigmoid()

        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))

        self.a = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.mu = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.sigma = nn.parameter.Parameter(torch.randn(out_features - 1))

    def forward(self, t):
        linear_feature = torch.matmul(t, self.w0) + self.b0
        gaussian_features = (torch.matmul(t, self.a) - self.mu) / (2 * self.sigma ** 2)
        out = torch.cat([linear_feature, gaussian_features], -1)
        out = self.sigmoid(out)
        return out / torch.sum(out, dim=-1)[..., None]
