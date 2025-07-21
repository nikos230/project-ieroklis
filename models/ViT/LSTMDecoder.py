import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, num_classes):
        super(LSTMDecoder, self).__init__()

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=0.5)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.norm = nn.LayerNorm(hidden_dim * 1)


    def forward(self, x):
        
        output, _ = self.lstm(x)
        out = output[:, -1, :] # last time step
        out = self.norm(out)
        out = self.classifier(out)

        return out
    


