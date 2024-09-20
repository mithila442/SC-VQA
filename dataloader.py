import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

class VQADataset(Dataset):
    def __init__(self, video_dir, mos_csv, saliency_dir, transform=None, num_frames=40, selected_videos=None):
        self.video_dir = video_dir
        self.saliency_dir = saliency_dir
        self.transform = transform
        self.num_frames = num_frames
        self.selected_videos = selected_videos  # Optional selected videos

        # Get the list of available video files in the directory
        self.available_videos = self._get_available_videos()

        # Load the CSV and filter based on available videos
        self.mos_df = pd.read_csv(mos_csv)
        self.valid_indices = self._filter_valid_videos()

    def __len__(self):
        return len(self.valid_indices)

    def _get_available_videos(self):
        # Get a set of all .mp4 files in the video directory
        available_videos = set(os.listdir(self.video_dir))
        # Filter available videos if selected_videos is provided
        if self.selected_videos is not None:
            available_videos = available_videos.intersection(self.selected_videos)
        return available_videos

    def _filter_valid_videos(self):
        valid_indices = []
        for idx in range(len(self.mos_df)):
            flickr_id = str(int(self.mos_df.iloc[idx]['flickr_id']))  # Convert to string without .0
            video_name = flickr_id + '.mp4'
            if video_name in self.available_videos:
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        # Map valid index back to original DataFrame index
        valid_idx = self.valid_indices[idx]

        flickr_id = str(int(self.mos_df.iloc[valid_idx]['flickr_id']))  # Adjust the column name if necessary
        video_name = flickr_id + '.mp4'
        mos_score = self.mos_df.iloc[valid_idx]['mos']  # Adjust the column name if necessary

        # Normalize MOS score between 0 and 1, assuming the range is [1, 5]
        mos_score = (mos_score - 1.0) / (5.0 - 1.0)

        # Full path to the video file
        video_path = os.path.join(self.video_dir, video_name)
        saliency_folder = os.path.join(self.saliency_dir,
                                       flickr_id)  # Assuming saliency maps are in folders named by flickr_id

        # Ensure the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        # Read video frames using OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the indices to sample 40 frames evenly across the video
        if frame_count >= self.num_frames:
            frame_indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        else:
            frame_indices = np.arange(0, frame_count)
            padding = self.num_frames - frame_count
            frame_indices = np.concatenate(
                [frame_indices, np.full(padding, frame_count - 1, dtype=int)])  # Pad with the last frame

        current_frame = 0
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break  # Stop if no more frames are available
            if current_frame in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame = Image.fromarray(frame)  # Convert to PIL Image
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            current_frame += 1
        cap.release()

        # Ensure the number of frames is exactly 40
        if len(frames) < self.num_frames:
            frames.extend([frames[-1]] * (self.num_frames - len(frames)))  # Pad with the last frame

        # Check if the saliency folder exists
        if not os.path.exists(saliency_folder):
            raise FileNotFoundError(f"Saliency folder does not exist: {saliency_folder}")

        # Load corresponding saliency maps
        saliency_maps = sorted([os.path.join(saliency_folder, img) for img in os.listdir(saliency_folder) if
                                img.endswith(('.jpg', '.jpeg', '.png'))])

        # Ensure exactly num_frames saliency maps
        if len(saliency_maps) >= self.num_frames:
            saliency_maps = saliency_maps[:self.num_frames]
        else:
            padding = self.num_frames - len(saliency_maps)
            saliency_maps.extend(
                [saliency_maps[-1]] * padding)  # Pad with the last saliency map if there are fewer maps

        # Ensure the number of frames and saliency maps match
        if len(frames) != len(saliency_maps):
            raise ValueError(
                f"Number of frames ({len(frames)}) and saliency maps ({len(saliency_maps)}) do not match for video {video_name}")

        frame_tensors = []
        for frame, saliency_map in zip(frames, saliency_maps):
            saliency = Image.open(saliency_map).convert('L')
            if self.transform:
                saliency = self.transform(saliency)
            # Concatenate the saliency map with the image along the channel dimension
            combined = torch.cat((frame, saliency), dim=0)  # Now combined has 4 channels: (3 RGB + 1 saliency)
            frame_tensors.append(combined)

        frames_tensor = torch.stack(frame_tensors)

        return frames_tensor, torch.tensor(mos_score, dtype=torch.float32), video_name


def collate_fn(batch):
    images = torch.cat([item[0].unsqueeze(0) for item in batch])
    saliency_maps = torch.cat([item[1].unsqueeze(0) for item in batch])
    mos_scores = torch.tensor([item[2] for item in batch])
    video_names = [item[3] for item in batch]

    return images, saliency_maps, mos_scores, video_names
