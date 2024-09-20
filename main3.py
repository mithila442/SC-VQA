from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models1 import UNetWithAttentionAnd3DConv
import torch.nn.functional as F
from utilities import *
from extract_frame import extract_frame
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import csv  # Import the CSV module

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    print(f"Normalizing tensor: min={min_val}, max={max_val}")  # Debug output to verify normalization
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
    return normalized_tensor

# Dataset class
class SaliencyDataset(Dataset):
    def __init__(self, root_dir, selected_indices, transform=None):
        self.root_dir = root_dir
        self.video_folders = [os.path.join(root_dir, f'{i:03d}') for i in range(1, 141)]
        self.selected_indices = selected_indices
        self.transform = transform

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        images = sorted([os.path.join(video_folder, 'extracted_frames', img) for img in
                         os.listdir(os.path.join(video_folder, 'extracted_frames')) if
                         img.endswith(('.jpg', '.jpeg', '.png'))])

        maps = sorted(
            [os.path.join(video_folder, 'map2', map_file) for map_file in os.listdir(os.path.join(video_folder, 'map2'))
             if map_file.endswith(('.jpg', '.jpeg', '.png'))])

        batch_images = []
        batch_maps = []
        for img_path, map_path in zip(images, maps):
            try:
                image = Image.open(img_path).convert('RGB')
                sal_map = Image.open(map_path).convert('L')

                if self.transform:
                    image = self.transform(image)
                    sal_map = self.transform(sal_map)

                batch_images.append(image)
                batch_maps.append(sal_map)

            except (UnidentifiedImageError, FileNotFoundError) as e:
                print(f"Error loading image {img_path} or {map_path}: {e}")
                continue

        if len(batch_images) == 0 or len(batch_maps) == 0:
            raise ValueError(f"No valid images or maps found in {video_folder}")

        batch_images = torch.stack(batch_images)
        batch_maps = torch.stack(batch_maps)

        # Ensure ground truth saliency maps are normalized (no inversion)
        batch_maps = normalize_tensor(batch_maps)

        return batch_images, batch_maps, video_folder


# Collate function for DataLoader
def collate_fn(batch):
    all_images = []
    all_maps = []
    video_folders = []

    for images, maps, video_folder in batch:
        all_images.append(images)
        all_maps.append(maps)
        video_folders.append(video_folder)

    max_length = max(images.size(0) for images in all_images)

    for i in range(len(all_images)):
        images = all_images[i]
        maps = all_maps[i]

        if images.size(0) < max_length:
            padding_size = max_length - images.size(0)
            padding_images = torch.zeros((padding_size, *images.shape[1:]), dtype=images.dtype)
            all_images[i] = torch.cat((images, padding_images), dim=0)

            padding_maps = torch.zeros((padding_size, *maps.shape[1:]), dtype=maps.dtype)
            all_maps[i] = torch.cat((maps, padding_maps), dim=0)

    all_images = torch.stack(all_images)
    all_maps = torch.stack(all_maps)

    return all_images, all_maps, video_folders


# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


# Function to train the model
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs, accumulation_steps, sub_batch_size,
                device, root_dir, expected_length):
    # Open a CSV file to write the loss for each epoch
    with open('epoch_loss.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])  # Write the header

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (images, maps, video_folders) in enumerate(dataloader):
                print(f"Epoch {epoch + 1}, Batch {i + 1}: Processing videos {video_folders}")
                images = images.to(device)
                maps = maps.to(device)
                outputs = []

                # Adjust the loop to match the expected sequence length
                for j in range(0, expected_length, sub_batch_size):
                    sub_images = images[:, j:j + sub_batch_size]
                    sub_maps = maps[:, j:j + sub_batch_size]

                    if sub_images.size(1) == 0:
                        continue

                    output = model(sub_images)
                    output = torch.sigmoid(output)  # Sigmoid for output
                    output = normalize_tensor(output)  # Normalize after sigmoid
                    sub_maps = normalize_tensor(sub_maps)  # Normalize maps as well
                    output = output.permute(0, 2, 1, 3, 4)  # Permute to [batch, channels, seq_length, height, width]
                    outputs.append(output)

                    loss = criterion(output, sub_maps)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    running_loss += loss.item() * sub_images.size(0)

                    # Print the loss for each sub-batch
                    print(
                        f"Epoch {epoch + 1}, Batch {i + 1}, Sub-batch {j // sub_batch_size + 1}: Loss = {loss.item():.4f}")

                if len(outputs) > 0:
                    outputs = torch.cat(outputs, dim=1)
                    outputs = outputs.permute(1, 0, 2, 3, 4)  # Permute to [seq_length, batch, channels, height, width]

                for k, video_folder in enumerate(video_folders):
                    output_folder = os.path.join(video_folder, 'saliency')
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    num_images = len(os.listdir(os.path.join(video_folder, 'extracted_frames')))
                    num_images = min(num_images, expected_length)  # Limit to expected length

                    for m in range(num_images):
                        img_name = f"{m:03d}.png"
                        try:
                            saliency_map = outputs[m, k].cpu().detach().numpy()
                        except IndexError as e:
                            print(f"IndexError: {e}, m: {m}, k: {k}, size of outputs: {outputs.size(0)}")
                            break

                        if len(saliency_map.shape) > 2:
                            saliency_map = saliency_map.squeeze()

                        saliency_map = (saliency_map * 255).astype(np.uint8)
                        if saliency_map.ndim == 3:
                            saliency_map = saliency_map[0]

                        original_image_path = os.path.join(video_folder, 'extracted_frames', img_name)
                        original_image = Image.open(original_image_path)
                        saliency_map_pil = Image.fromarray(saliency_map)
                        saliency_map_pil.save(os.path.join(output_folder, img_name))

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
            writer.writerow([epoch + 1, epoch_loss])  # Write the epoch and loss to the CSV file
            file.flush()
            scheduler.step()
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, epoch, epoch_loss, path=f'checkpoint_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(torch.cuda.device_count(), "GPUs are available.")
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))

    phase = 'train'
    if phase == 'train':
        stateful = False
    else:
        stateful = True

    if phase == "train":
        num_epochs = 200
        learning_rate = 5e-3
        batch_size = 2  # Number of videos processed simultaneously
        accumulation_steps = 4
        sub_batch_size = 4  # Number of frames processed at once per sub-batch
        expected_length = 40  # Number of seconds per video

        train_data_folder = './video_data/'

        data_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.ToTensor(),
        ])

        # Hyperparameters for frame extraction
        size = 224  # Resize dimension for frames
        video_length_min = 40  # Total number of frames extracted (8 seconds * 5 frames per second)
        frames_per_second = 5  # Number of frames to extract per second

        videos_dir = os.path.join(train_data_folder, 'videos')

        selected_indices = []

        for i in range(1, 141):
            video_name = f'{i:03d}.AVI'
            save_folder = os.path.join(train_data_folder, f'{i:03d}', 'extracted_frames')
            os.makedirs(save_folder, exist_ok=True)

            # Use the updated extract_frame function with the frames_per_second parameter
            indices = extract_frame(videos_dir, video_name, size, save_folder, video_length_min,
                                    frames_per_second=frames_per_second)

            selected_indices.append(indices)
            print(f'Extracted frames and maps from {video_name} to {save_folder}')

        # Create dataset and dataloader
        train_dataset = SaliencyDataset(train_data_folder, selected_indices, transform=data_transforms)
        print(f"Found {len(train_dataset)} video files in {train_data_folder}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                  collate_fn=collate_fn)

        # Initialize the model and training process
        model = UNetWithAttentionAnd3DConv(in_channels=3, out_channels=1)
        model=nn.DataParallel(model).to(device)
        criterion = CombinedLoss(alpha1=0.01, alpha2=0.1)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) #momentum=0.2)
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)

        # Train the model
        train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=num_epochs,
                    accumulation_steps=accumulation_steps, sub_batch_size=sub_batch_size, device=device,
                    root_dir=train_data_folder, expected_length=expected_length)
