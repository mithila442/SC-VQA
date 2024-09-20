import os
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np
from models1 import UNetWithAttentionAnd3DConv

class SaliencyMapGenerator:
    def __init__(self, model_checkpoint, device, num_frames=150):
        self.device = device
        self.num_frames = num_frames
        self.model = UNetWithAttentionAnd3DConv(in_channels=3, out_channels=1).to(self.device)

        print(f"Loading checkpoint from {model_checkpoint}...")

        try:
            checkpoint = torch.load(model_checkpoint, map_location=self.device)

            # Adjust state_dict if it was saved with DataParallel
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')  # remove `module.` prefix
                new_state_dict[new_key] = value

            self.model.load_state_dict(new_state_dict)
            print("Model loaded and moved to device.")
        except torch.cuda.OutOfMemoryError as e:
            print("CUDA out of memory. Attempting to load checkpoint on CPU...")
            checkpoint = torch.load(model_checkpoint, map_location=torch.device('cpu'))

            # Adjust state_dict if it was saved with DataParallel
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')  # remove `module.` prefix
                new_state_dict[new_key] = value

            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device)
            print("Model loaded on CPU and moved to device.")

        self.model.eval()

    def generate_saliency_map(self, video_frame):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Resize the input video frame to 224x398
                input_tensor = transforms.Compose([
                    transforms.Resize((224, 398)),
                    transforms.ToTensor()
                ])(video_frame).unsqueeze(0).to(self.device)

                # Add a sequence dimension to the input tensor
                input_tensor = input_tensor.unsqueeze(1)  # Add sequence length of 1

                saliency_map = self.model(input_tensor)

                # Remove batch and sequence dimensions to make it 2D or 3D
                saliency_map = saliency_map.squeeze(0).squeeze(0).cpu()

                # Convert to PIL Image
                saliency_map = transforms.ToPILImage()(saliency_map)
                saliency_map = saliency_map.convert("L")

                # Apply contrast enhancement
                enhancer = ImageEnhance.Contrast(saliency_map)
                saliency_map = enhancer.enhance(10.0)  # Adjust this factor as needed
                #saliency_map = ImageOps.invert(saliency_map)

        return saliency_map

    def process_video(self, video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine the frame indices to sample
        if total_frames >= self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            frame_indices = np.arange(0, total_frames)
            padding = self.num_frames - total_frames
            frame_indices = np.concatenate([frame_indices, np.full(padding, total_frames - 1, dtype=int)])

        print(f"Sampling frame indices: {frame_indices}")

        os.makedirs(output_dir, exist_ok=True)
        frame_idx = 0
        processed_frame_count = 0  # Counter for processed frames

        while True:
            ret, frame = cap.read()
            if not ret or processed_frame_count >= self.num_frames:
                break  # End of video or reached the desired number of frames

            if frame_idx in frame_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                saliency_map = self.generate_saliency_map(frame_pil)
                output_path = os.path.join(output_dir, f"{processed_frame_count:04d}.png")
                saliency_map.save(output_path)
                print(f"Processed frame {frame_idx} and saved saliency map to {output_path}")
                processed_frame_count += 1

            frame_idx += 1

        cap.release()

if __name__ == "__main__":
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model_checkpoint = './checkpoint_epoch_181.pth'  # Change this to your actual checkpoint file path

    print("Initializing SaliencyMapGenerator...")
    generator = SaliencyMapGenerator(
        model_checkpoint=model_checkpoint,
        device=device,
        num_frames=40  # Change this as needed
    )

    video_root_dir = './KoNViD_1k/KoNViD_1k_videos'  # Root directory containing videos
    output_root_dir = './KoNViD_1k_saliency_maps'  # Root directory for saving saliency maps

    video_files = sorted([f for f in os.listdir(video_root_dir) if f.endswith('.mp4')])

    print(f"Found video files: {video_files}")

    for video_file in video_files:
        video_path = os.path.join(video_root_dir, video_file)
        output_dir = os.path.join(output_root_dir, video_file.replace('.mp4', ''))
        print(f"Processing video {video_file}...")
        generator.process_video(video_path, output_dir)
        print(f"Finished processing video {video_file}")
