import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)  # Concatenate embeddings
        z = F.normalize(z, p=2, dim=1)

        sim_matrix = torch.mm(z, z.T) / self.temperature  # Similarity matrix
        sim_matrix = torch.exp(sim_matrix)

        # Mask to remove self-similarities
        mask = (~torch.eye(sim_matrix.size(0), device=self.device).bool()).float()

        # Numerator: positive pairs
        batch_size = z_i.size(0)
        if sim_matrix.size(0) != 2 * batch_size:
            raise ValueError(f"Mismatch: sim_matrix size {sim_matrix.size(0)}, expected {2 * batch_size}")

        sim_pos = torch.cat([torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)])

        # Denominator: all pairs excluding self
        sim_den = torch.sum(sim_matrix * mask, dim=1)

        loss = -torch.mean(torch.log(sim_pos / sim_den))

        return loss


class LSTMTemporalAttention(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, output_dim=64, dropout=0.3):
        super(LSTMTemporalAttention, self).__init__()

        self.conv2d_1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1)
        nn.init.kaiming_normal_(self.conv2d_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv2d_1.bias, 0)

        self.bn1 = nn.BatchNorm2d(num_features=hidden_dim, eps=1e-5, momentum=0.1)


        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.lstm = nn.LSTM(input_size=hidden_dim * 8 * 8, hidden_size=hidden_dim, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        self.fc = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

        self.attention_fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.attention_fc.weight)
        nn.init.constant_(self.attention_fc.bias, 0)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=False)

    def forward_one(self, x):
        batch_size, channels, frames, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, frames, channels, height, width]
        x = x.reshape(batch_size * frames, channels, height, width)

        x = self.conv2d_1(x)
        if torch.isnan(x).any():
            print("NaN detected after Conv2D")
            exit()
        x = self.bn1(x)
        if torch.isnan(x).any():
            print("NaN detected after GroupNorm")
            exit()
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(batch_size, frames, -1)
        lstm_out, _ = self.lstm(x)
        if torch.isnan(lstm_out).any():
            print("NaN detected after LSTM")
            exit()

        attention_scores = self.attention_fc(lstm_out)
        attention_weights = torch.softmax(attention_scores.squeeze(-1), dim=1).unsqueeze(-1)
        weighted_x = lstm_out * attention_weights

        x = weighted_x.sum(dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        if torch.isnan(x).any():
            print("NaN detected after final projection (fc)")
            exit()

        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
