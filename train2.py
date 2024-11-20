import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import csv
from contrast import LSTMTemporalAttention, NT_Xent
from dataloader import VQADataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.transforms import ToPILImage
from torch.utils.data.distributed import DistributedSampler

# Set anomaly detection for debugging in-place operations
#torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.enabled = False

class NeuralNetworkRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(NeuralNetworkRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)

class SiameseNetworkTrainer:
    def __init__(self, video_dir, mos_csv, saliency_dir, device, dropout=0.5, alpha=0.01):
        self.device = device
        self.video_dir = video_dir
        self.alpha = alpha  # Weight for combining regression and contrastive loss
        self.best_val_loss = float('inf')

        # Data transformation for contrastive learning
        self.data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor()
        ])

        # Load VQA dataset
        self.dataset = VQADataset(video_dir, mos_csv, saliency_dir=saliency_dir, transform=self.data_transforms, num_frames=8)

        # Select only the first 500 videos
        total_videos = min(len(self.dataset), 500)  # Ensure we don't exceed the dataset size
        selected_indices = list(range(total_videos))  # Use the first 500 videos
        train_indices, val_indices = train_test_split(selected_indices, test_size=0.2, random_state=953)  # 80/20 split
        np.save('val_indices.npy', val_indices)

        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)

        # Split the val_dataset into validation and test sets
        val_size = len(self.val_dataset) // 2
        test_size = len(self.val_dataset) - val_size  # Ensures the total length is preserved
        val_subset, test_subset = random_split(self.val_dataset, [val_size, test_size])

        # Save test_indices.npy
        np.save('test_indices.npy', test_subset.indices)

        # Create DataLoader for each subset
        self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
        self.val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)
        self.test_loader = DataLoader(test_subset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)

        # Initialize embedding model and regressor
        in_channels, hidden_dim, output_dim = 3, 256, 128
        self.model = LSTMTemporalAttention(in_channels, hidden_dim, output_dim, dropout=0.6).to(self.device)
        self.regressor = NeuralNetworkRegressor(input_dim=output_dim).to(self.device)

        # Define loss functions
        self.contrastive_criterion = NT_Xent(batch_size=self.train_loader.batch_size, temperature=0.1, device=self.device)
        self.regression_criterion = nn.MSELoss()

        # Optimizers
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.regressor.parameters()), lr=0.0001)

        # Cosine Annealing Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-8)

        # CSV file to log losses
        self.loss_log_file = 'loss_log.csv'
        with open(self.loss_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Contrastive Loss', 'Regression Loss', 'Combined Loss', 'Val Contrastive Loss', 'Val Regression Loss'])

            
            
    def _augment_frames(self, frames):
        """
        Apply data augmentation to create two distinct views of each video batch for contrastive learning.
        Ensures outputs are consistent with the device (self.device).
        Assumes input 'frames' shape is [batch_size, channels, num_frames, height, width].
        """
        to_pil = ToPILImage()  # Initialize the conversion to PIL
        batch_size, channels, num_frames, height, width = frames.size()

        # Move frames to the device for processing
        frames = frames.to(self.device)

        frames = frames.permute(0, 2, 1, 3, 4)  # Change to [batch_size, num_frames, channels, height, width]

        view1, view2 = [], []
        for i in range(batch_size):
            for j in range(num_frames):
                frame = frames[i, j]  # Get each frame as [channels, height, width]
                frame_pil = to_pil(frame.cpu())  # Convert to PIL image (requires CPU tensor)

                # Apply augmentations
                augmented_view1 = self.data_transforms(frame_pil)
                augmented_view2 = self.data_transforms(frame_pil)

                view1.append(augmented_view1)
                view2.append(augmented_view2)

        # Stack and reshape back to [batch_size, num_frames, channels, height, width]
        view1 = torch.stack(view1).view(batch_size, num_frames, channels, height, width)
        view2 = torch.stack(view2).view(batch_size, num_frames, channels, height, width)

        # Permute to [batch_size, channels, num_frames, height, width] for model compatibility
        view1 = view1.permute(0, 2, 1, 3, 4).to(self.device)  # Move to GPU
        view2 = view2.permute(0, 2, 1, 3, 4).to(self.device)  # Move to GPU

        return view1, view2

    def validate(self):
        self.model.eval()
        self.regressor.eval()
        val_contrastive_loss, val_regression_loss = 0.0, 0.0

        with torch.no_grad():
            for frames, mos_labels, _ in self.val_loader:
                frames, mos_labels = frames.to(self.device), mos_labels.to(self.device)

                # Augment frames
                view1, view2 = self._augment_frames(frames)

                # Forward pass for contrastive embeddings
                output1, output2 = self.model.forward_one(view1), self.model.forward_one(view2)

                # Compute contrastive loss
                contrastive_loss = self.contrastive_criterion(output1, output2)

                # Regression loss
                avg_embedding = (output1 + output2) / 2
                predicted_score = self.regressor(avg_embedding).view(-1)
                regression_loss = self.regression_criterion(predicted_score, mos_labels)

                val_contrastive_loss += contrastive_loss.item()
                val_regression_loss += regression_loss.item()

        avg_val_contrastive_loss = val_contrastive_loss / len(self.val_loader)
        avg_val_regression_loss = val_regression_loss / len(self.val_loader)
        return avg_val_contrastive_loss, avg_val_regression_loss


    def train_contrastive(self, num_epochs=15, rank=0, accumulation_steps=4):
        """
        Train the model using a multi-task framework combining contrastive loss (self-supervised)
        and regression loss (supervised).

        Args:
            num_epochs (int): Number of epochs to train.
            rank (int): Rank of the process (for distributed training).
            accumulation_steps (int): Number of steps to accumulate gradients before optimizer step.
        """
        for epoch in range(num_epochs):
            # Print the current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            #print(f"Epoch [{epoch + 1}/{num_epochs}] - Current LR: {current_lr}")

            self.model.train()
            self.regressor.train()

            running_contrastive_loss = 0.0
            running_regression_loss = 0.0
            running_combined_loss = 0.0
            self.optimizer.zero_grad()

            for batch_idx, (frames, mos_labels, _) in enumerate(self.train_loader):  # Ignore dist_labels
                # Move frames and labels to device
                frames, mos_labels = frames.to(self.device), mos_labels.to(self.device)

                # Augment frames and forward pass
                view1, view2 = self._augment_frames(frames)
                z_i, z_j = self.model.forward_one(view1), self.model.forward_one(view2)

                # Compute contrastive loss (self-supervised)
                contrastive_loss = self.contrastive_criterion(z_i, z_j)

                # Compute regression loss (supervised MOS prediction)
                avg_embedding = (z_i + z_j) / 2  # Average of embeddings from both views
                predicted_score = self.regressor(avg_embedding).squeeze()
                regression_loss = self.regression_criterion(predicted_score, mos_labels)

                # Combine losses with a weight factor (self.alpha)
                combined_loss = self.alpha* contrastive_loss + regression_loss
                combined_loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=5.0)

                # Accumulate gradients and optimizer step
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update running losses
                running_contrastive_loss += contrastive_loss.item()
                running_regression_loss += regression_loss.item()
                running_combined_loss += combined_loss.item()

            # Average epoch losses
            epoch_contrastive_loss = running_contrastive_loss / len(self.train_loader)
            epoch_regression_loss = running_regression_loss / len(self.train_loader)
            epoch_combined_loss = running_combined_loss / len(self.train_loader)

            # Validation
            if rank == 0:
                avg_val_contrastive_loss, avg_val_regression_loss = self.validate()
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Contrastive Loss: {epoch_contrastive_loss:.4f}, "
                    f"Regression Loss: {epoch_regression_loss:.4f}, "
                    f"Combined Loss: {epoch_combined_loss:.4f}, "
                    f"Val Contrastive Loss: {avg_val_contrastive_loss:.4f}, "
                    f"Val Regression Loss: {avg_val_regression_loss:.4f}, "
                    f"Current LR: {current_lr:.4f}"
                )

                # Save best model checkpoint
                avg_combined_val_loss = avg_val_contrastive_loss + self.alpha * avg_val_regression_loss
                if avg_combined_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_combined_val_loss
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "regressor_state_dict": self.regressor.state_dict(),
                        },
                        f"checkpoint_epoch_{epoch + 1}_rank_{rank}.pth",
                    )
                    print(f"Checkpoint saved for epoch {epoch + 1}, rank {rank}")

            # Step scheduler
            self.scheduler.step()




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # Initialize the process group with the NCCL backend
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    trainer = SiameseNetworkTrainer(
        video_dir='./KoNViD_1k/KoNViD_1k_videos',
        mos_csv='./KoNViD_1k/KoNViD_1k_mos.csv',
        saliency_dir='./KoNViD_1k_saliency_maps/',
        device=device,
        dropout=0.5,
        alpha=0.1
    )

    trainer.train_contrastive(num_epochs=2000, rank=rank)
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
