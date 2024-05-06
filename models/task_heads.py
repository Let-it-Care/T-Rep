import torch
import torch.nn as nn


class TembedDivPredHead(nn.Module):

    def __init__(self, in_features, out_features, hidden_features=64, dropout=0.1):
        super(TembedDivPredHead, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            self.relu,
            nn.Linear(hidden_features, out_features),
            self.relu,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.fc(x))

class TembedKLPredHeadLinear(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.1):
        super(TembedKLPredHeadLinear, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            self.sigmoid,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.fc(x))


class TembedCondPredHead(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, dropout=0.1):
        super(TembedCondPredHead, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            self.relu,
            nn.Linear(hidden_features[0], out_features),
            self.relu,
            self.dropout,

        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.fc(x)
