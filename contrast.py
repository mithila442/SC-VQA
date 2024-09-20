import torch
import torch.nn as nn

class Conv3DSiameseNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, output_dim=128, kernel_size=(3, 3, 3), dropout=0.5):
        super(Conv3DSiameseNetwork, self).__init__()

        # First Conv3D layer to handle spatiotemporal video data
        self.conv3d_1 = nn.Conv3d(in_channels, hidden_dim, kernel_size=kernel_size, stride=(1, 1, 1), padding=1)
        self.relu = nn.ReLU()

        # Add a second Conv3D layer to increase the model's capacity
        self.conv3d_2 = nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=kernel_size, stride=(1, 1, 1), padding=1)

        # Apply adaptive pooling to reduce the dimensionality after Conv3D layers
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))  # Reduce output size to (batch_size, hidden_dim * 2, 4, 4, 4)

        # Dropout layer for regularization (increase dropout to prevent overfitting)
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer based on the reduced size from the pooling
        self.fc = nn.Linear(hidden_dim * 2 * 4 * 4 * 4, output_dim)  # Based on output from conv3d_2

    def forward_one(self, x):
        # x is (batch_size, num_frames, channels, height, width)
        # Conv3D expects (batch_size, channels, num_frames, height, width), so we permute
        x = x.permute(0, 2, 1, 3, 4)  # Change to (batch_size, channels, num_frames, height, width)

        # Pass through the first Conv3D layer
        conv_out_1 = self.relu(self.conv3d_1(x))

        # Pass through the second Conv3D layer to capture more complex patterns
        conv_out_2 = self.relu(self.conv3d_2(conv_out_1))

        # Apply adaptive pooling to reduce dimensionality
        pooled_out = self.pool(conv_out_2)

        # Apply dropout to regularize the model
        pooled_out = self.dropout(pooled_out)

        # Flatten the output for the fully connected layer
        batch_size = pooled_out.size(0)
        pooled_out = pooled_out.view(batch_size, -1)  # Flatten all dimensions except batch size

        # Pass through the fully connected layer
        out = self.fc(pooled_out)
        return out

    def forward(self, input1, input2):
        # Get outputs for the two inputs
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


# Fixed Contrastive Loss with a constant margin
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.7):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, output1, output2, label):
        # Ensure label is float tensor
        label = label.float()

        # 1. Euclidean Loss (labels {1, 0}):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        euclidean_loss_similar = label * torch.pow(euclidean_distance, 2)
        euclidean_loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        euclidean_loss = torch.mean(euclidean_loss_similar + euclidean_loss_dissimilar)

        # 2. Cosine Loss (requires labels {1, -1}):
        # Convert label {1, 0} -> {1, -1}
        cosine_label = 2 * label - 1  # 1 becomes 1, 0 becomes -1
        cosine_loss = self.cosine_loss(output1, output2, cosine_label)

        # 3. Combined Loss (weighted sum of both losses)
        combined_loss = self.alpha * euclidean_loss + (1 - self.alpha) * cosine_loss

        return combined_loss
