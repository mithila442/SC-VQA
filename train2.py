import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import numpy as np
import csv
from contrast import Conv3DSiameseNetwork, ContrastiveLoss  # Import updated Conv3D-based Siamese network
from dataloader import VQADataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
from PIL import Image

# Neural Network Regressor for MOS prediction
class NeuralNetworkRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(NeuralNetworkRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))  # Apply dropout before final layer
        x = self.fc3(x)
        return x


# Define the training class for the Conv3D Siamese Network model with neural network regressor
class SiameseNetworkTrainer:
    def __init__(self, video_dir, mos_csv, saliency_dir, device, beta=0.7, dropout=0.5, save_extracted_frames=True):
        self.device = device  # The device on which the model will run (CPU or GPU)
        self.beta = beta  # The weight for combining contrastive loss and regression loss
        self.save_extracted_frames = save_extracted_frames  # Whether to save the extracted frames or not

        # Get a random subset of 150 video files from the directory
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        random_selected_videos = random.sample(video_files, 200)

        # Data transformations applied to each frame in the dataset (resize and convert to tensor)
        data_transforms = transforms.Compose([
            transforms.Resize((224, 398)),  # Resize frames to 224x398
            transforms.ToTensor(),  # Convert image to PyTorch tensor
        ])

        # Load the VQA dataset with frames and saliency maps, but only for the selected videos
        self.dataset = VQADataset(
            video_dir,
            mos_csv,
            saliency_dir=saliency_dir,
            transform=data_transforms,
            selected_videos=random_selected_videos,
            num_frames=40  # Set the number of frames to extract
        )

        # Ensure extracted frames are saved if `save_extracted_frames` is True
        if self.save_extracted_frames:
            self._save_frames_for_selected_videos(random_selected_videos, data_transforms)

        # Split dataset into training and evaluation sets (80% training, 20% evaluation)
        train_indices, eval_indices = train_test_split(range(len(self.dataset)), test_size=0.2, random_state=953)

        # Create subsets for training and evaluation
        self.train_dataset = Subset(self.dataset, train_indices)
        self.eval_dataset = Subset(self.dataset, eval_indices)

        # Create data loaders for training and evaluation
        self.train_loader = DataLoader(self.train_dataset, batch_size=2, shuffle=True, num_workers=4)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=1, shuffle=False, num_workers=4)

        # Initialize the Conv3D-based Siamese network (with dropout) and move it to the specified device
        in_channels = 3  # RGB channels
        hidden_dim = 64  # Hidden dimension for Conv3D
        output_dim = 128  # Output embedding dimension
        self.model = Conv3DSiameseNetwork(in_channels, hidden_dim, output_dim, dropout=dropout).to(self.device)

        # Contrastive loss (for contrastive learning)
        self.contrastive_criterion = ContrastiveLoss(margin=1.5, alpha=0.7)  # You can use Cosine Embedding Loss for similarity comparison

        # Optimizer (Adam) and weight decay for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-4)

        # Add Cosine Annealing LR scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)

        # Initialize the neural network regressor
        self.regressor = NeuralNetworkRegressor(input_dim=output_dim).to(self.device)

        # Optimizer for the regressor with L2 regularization (weight decay)
        self.regressor_optimizer = optim.Adam(self.regressor.parameters(), lr=0.001, weight_decay=1e-4)

        # Create a CSV file to log losses
        self.loss_log_file = 'loss_log.csv'
        with open(self.loss_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Contrastive Loss', 'Regression Loss', 'Combined Loss'])  # Header for the CSV file

    def _save_frames_for_selected_videos(self, selected_videos, data_transforms):
        """
        Helper function to extract and save frames for the selected videos.
        """
        for video in selected_videos:
            # Load the video and extract the frames using OpenCV
            video_path = os.path.join(self.dataset.video_dir, video)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the indices to sample 40 frames evenly across the video
            if frame_count >= 40:
                frame_indices = np.linspace(0, frame_count - 1, 40, dtype=int)
            else:
                frame_indices = np.arange(0, frame_count)
                padding = 40 - frame_count
                frame_indices = np.concatenate(
                    [frame_indices, np.full(padding, frame_count - 1, dtype=int)])  # Pad with the last frame

            current_frame = 0
            extracted_frames_dir = f'./extracted_frames/{video.replace(".mp4", "")}'
            os.makedirs(extracted_frames_dir, exist_ok=True)

            while current_frame in frame_indices:
                ret, frame = cap.read()
                if not ret:
                    break  # Stop if no more frames are available
                if current_frame in frame_indices:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    pil_frame = Image.fromarray(frame)  # Convert to PIL Image

                    # Save the frame before applying transformation
                    frame_save_path = os.path.join(extracted_frames_dir, f'frame_{current_frame:04d}.png')
                    pil_frame.save(frame_save_path)  # Save PIL image

                    # Apply transformation if necessary for further processing
                    if data_transforms:
                        frame_tensor = data_transforms(pil_frame)

                current_frame += 1
            cap.release()

    # Training function for contrastive learning and MOS prediction with a neural network regressor
    def train_contrastive(self, num_epochs=15):
        self.model.train()  # Set the model to training mode
        self.regressor.train()  # Set the regressor to training mode
        for epoch in range(num_epochs):
            running_contrastive_loss = 0.0
            running_regression_loss = 0.0
            running_combined_loss = 0.0

            for batch_idx, (frames_with_saliency, mos_labels, _) in enumerate(self.train_loader):
                # Move frames and MOS labels to the device (GPU/CPU)
                frames_with_saliency = frames_with_saliency.to(self.device)
                mos_labels = mos_labels.to(self.device)

                # Separate frames (RGB) and saliency maps
                frames = frames_with_saliency[:, :, :3, :, :]  # Extract RGB channels

                self.optimizer.zero_grad()  # Zero the gradients for the Siamese network
                self.regressor_optimizer.zero_grad()  # Zero the gradients for the regressor

                batch_size = frames.size(0)

                # Forward pass for contrastive learning
                positive_loss = 0.0
                negative_loss = 0.0
                positive_count = 0
                negative_count = 0

                # Create pairs within the batch for contrastive learning
                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        # Generate labels for similar (1) and dissimilar (-1) pairs based on MOS scores
                        label = 1 if abs(mos_labels[i] - mos_labels[j]) < 0.08 else 0
                        label = torch.tensor([label], dtype=torch.float32).to(self.device)

                        # Forward pass (pairwise for contrastive loss)
                        output1, output2 = self.model(frames[i].unsqueeze(0), frames[j].unsqueeze(0))

                        # Calculate combined contrastive loss (both Euclidean and Cosine) automatically
                        loss = self.contrastive_criterion(output1, output2, label)

                        # Class-Averaging: separate positive and negative pairs
                        if label == 1:
                            positive_loss += loss
                            positive_count += 1
                        else:
                            negative_loss += loss
                            negative_count += 1

                # Average out positive and negative losses
                if positive_count > 0:
                    positive_loss /= positive_count
                if negative_count > 0:
                    negative_loss /= negative_count

                # Combine both losses with class-averaging
                contrastive_loss = (positive_loss + negative_loss) / 2

                # Forward pass for MOS prediction using the neural network regressor
                with torch.no_grad():
                    embeddings = [self.model.forward_one(frames[i].unsqueeze(0)).view(-1) for i in range(batch_size)]
                    embeddings = torch.stack(embeddings)

                # Pass the embeddings through the regressor and compute MSE loss
                predicted_mos = self.regressor(embeddings)
                regression_loss = nn.MSELoss()(predicted_mos, mos_labels.unsqueeze(1))

                # Combined Loss: contrastive loss + alpha * regression loss
                combined_loss = (1-self.beta)*contrastive_loss + self.beta*regression_loss

                # Backpropagation for combined loss (single backpropagation step)
                combined_loss.backward()

                # Update the model parameters
                self.regressor_optimizer.step()
                self.optimizer.step()

                # Adjust the learning rate using the scheduler after optimizer step
                self.scheduler.step()

                # Accumulate running losses separately
                running_contrastive_loss += contrastive_loss.item()
                running_regression_loss += regression_loss.item()
                running_combined_loss += combined_loss.item()

            # Log the losses for the current epoch
            epoch_contrastive_loss = running_contrastive_loss / len(self.train_loader)
            epoch_regression_loss = running_regression_loss / len(self.train_loader)
            epoch_combined_loss = running_combined_loss / len(self.train_loader)

            # Save the losses into the CSV file
            with open(self.loss_log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, epoch_contrastive_loss, epoch_regression_loss, epoch_combined_loss])

            # Optionally print the losses for the current epoch
            print(f'Epoch [{epoch + 1}/{num_epochs}], Contrastive Loss: {epoch_contrastive_loss:.4f}, Regression Loss: {epoch_regression_loss:.4f}, Combined Loss: {epoch_combined_loss:.4f}')


    # Evaluation function using the trained regressor
    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        self.regressor.eval()  # Set the regressor to evaluation mode
        embeddings_list = []  # To store the embeddings for all videos
        mos_labels_list = []  # To store the true MOS labels
        mos_predictions_list = []  # To store the predicted MOS values

        with torch.no_grad():  # No need to track gradients during evaluation
            for frames_with_saliency, mos_labels, _ in self.eval_loader:
                frames_with_saliency = frames_with_saliency.to(self.device)
                frames = frames_with_saliency[:, :, :3, :, :]  # Extract RGB channels

                # Get the embeddings from the model using contrastive learning
                embeddings = torch.stack([self.model.forward_one(frames[i].unsqueeze(0)).view(-1) for i in range(frames.size(0))])

                # Predict MOS values using the embeddings (from contrastive learning) via the neural network regressor
                mos_predictions = self.regressor(embeddings).cpu().numpy()

                # Store the predictions and corresponding labels
                mos_predictions_list.extend(mos_predictions)  # Extend instead of append to flatten batches
                mos_labels_list.extend(mos_labels.cpu().numpy())  # Ensure both lists are flattened

                # Print the predicted score vs actual score for each video
                for i in range(len(mos_predictions)):
                    predicted = float(mos_predictions[i].item())  # Convert to float
                    actual = float(mos_labels[i].item())
                    print(f"Predicted MOS: {predicted:.4f}, Actual MOS: {actual:.4f}")

        # Convert lists to arrays
        mos_labels_array = np.hstack(mos_labels_list)  # Flatten the list of arrays into a single array
        mos_predictions_array = np.hstack(mos_predictions_list)  # Flatten predictions similarly

        # Ensure both arrays have the same length
        if len(mos_labels_array) != len(mos_predictions_array):
            print(f"Error: Mismatch in lengths. MOS labels: {len(mos_labels_array)}, Predictions: {len(mos_predictions_array)}")
            return

        # Calculate Pearson Linear Correlation Coefficient (PLCC) and Spearman Rank-Order Correlation Coefficient (SROCC)
        plcc = pearsonr(mos_labels_array, mos_predictions_array)[0]
        srocc = spearmanr(mos_labels_array, mos_predictions_array)[0]

        # Print evaluation results
        print(f"Evaluation PLCC: {plcc:.4f}, SROCC: {srocc:.4f}")


if __name__ == "__main__":
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the trainer class with paths to your video directory, MOS CSV, and saliency maps directory
    trainer = SiameseNetworkTrainer(
        video_dir='./KoNViD_1k/KoNViD_1k_videos',  # Path to video directory
        mos_csv='./KoNViD_1k/KoNViD_1k_mos.csv',  # Path to MOS CSV file
        saliency_dir='./KoNViD_1k_saliency_maps/',  # Path to saliency maps directory
        device=device,  # Device to run the model on (CPU or GPU)
        beta=0.6,  # Alpha value for weighting the regression loss
        dropout=0.5  # Dropout rate
    )

    # Train the model using contrastive learning and also train the neural network regressor
    trainer.train_contrastive(num_epochs=200)

    # Evaluate the model using the trained regressor and calculate PLCC and SROCC
    trainer.evaluate()
