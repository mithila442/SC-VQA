# Saliency-Driven Contrastive Video Quality Assessment (SC-VQA) Model

This project implements the **Saliency-Driven Contrastive Video Quality Assessment (SC-VQA) Model**, a robust architecture designed to predict the perceptual quality of video sequences. By leveraging saliency maps generated from eye-tracking data, the model focuses on the most visually significant regions of video frames, enhancing its ability to align with human perception. 

## Project Overview

- **Purpose**: To evaluate the quality of video sequences by predicting Mean Opinion Scores (MOS), aligning with human perception of video quality through saliency-driven insights.
- **Saliency Integration**: Pretrained on eye-tracking datasets, the model generates saliency maps that guide the attention mechanism, focusing on crucial areas that impact perceived video quality.
- **Technologies Used**: Python, PyTorch, Conv3D, DeConv3D, Contrastive Learning.

## Model Architecture

1. **Encoder and Decoder Blocks**: 
   - Utilizes Conv3D layers for feature extraction and DeConv3D layers for reconstructing saliency maps, highlighting important regions of the video.

2. **Attention Mechanisms**:
   - Integrates saliency maps to focus on visually significant areas, enhancing the model's predictive capabilities.

3. **Contrastive Learning**:
   - Learns to differentiate between high and low-quality video segments using positive and negative pairs of embeddings.

4. **Neural Network Regressor**:
   - Predicts the video quality score (MOS) based on embeddings generated from the contrastive learning module.

5. **Pretraining and Saliency Integration**:
   - Pretrained on the eye-tracking dataset DHF1K to generate saliency maps, which are further used to refine attention on VQA datasets.

## Key Files

**contrast.py**: Implements the contrastive learning framework to enhance the differentiation of video quality through positive and negative pairs.
- **dataloader.py**: Manages data loading and preprocessing, ensuring frames are prepared correctly for training and inference.
- **extract_frame.py**: Extracts frames from video sequences for input into the encoder.
- **inference.py**: Runs the pre-trained model on new video quality data to predict quality scores.
- **main3.py**: Trains the saliency model using eye-tracking datasets, focusing on generating attention maps for the VQA task.
- **models1.py**: Defines the architecture, including encoder, decoder, attention modules, and neural network regressor.
- **train2.py**: Handles the training of the VQA model, combining contrastive and regression losses to optimize quality prediction.
- **utilities.py**: Contains utility functions for metrics calculation, data manipulation, and other auxiliary tasks.

## How to Use

1. **Setup**: Ensure all dependencies are installed as per the requirements.
2. **Saliency Training**: Use `main3.py` to train the saliency model on your eye-tracking dataset.
3. **Inference**: Use `inference.py` to apply the pre-trained model to new video data.
4. **Quality Assessment Training**: Use `train2.py` to train the VQA model based on contrastive loss and regression loss.

## Authors

- Mayesha Maliha Rahman Mithila, Mylene C.Q. Farias

## License

This project is licensed under the MIT License.

