from PIL import Image
import numpy as np
import scipy.ndimage
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

def kl_divergence_loss(y_pred, y_true):
    y_pred = F.log_softmax(y_pred, dim=1)
    y_true = F.softmax(y_true, dim=1)
    return F.kl_div(y_pred, y_true, reduction='batchmean')


def correlation_coefficient_loss(y_pred, y_true):
    y_pred_mean = torch.mean(y_pred, dim=(1, 2, 3), keepdim=True)
    y_true_mean = torch.mean(y_true, dim=(1, 2, 3), keepdim=True)
    y_pred_centered = y_pred - y_pred_mean
    y_true_centered = y_true - y_true_mean
    correlation = torch.sum(y_pred_centered * y_true_centered, dim=(1, 2, 3))
    std_pred = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=(1, 2, 3)) + 1e-6)
    std_true = torch.sqrt(torch.sum(y_true_centered ** 2, dim=(1, 2, 3)) + 1e-6)
    return -correlation / (std_pred * std_true + 1e-6)


class CombinedLoss(nn.Module):
    def __init__(self, alpha1=0.1, alpha2=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, y_pred, y_true):
        l_kl = kl_divergence_loss(y_pred, y_true)
        l_cc = correlation_coefficient_loss(y_pred, y_true)
        loss = self.alpha1 * l_kl + l_cc
        return loss.mean()

def imread(path):
    return np.array(Image.open(path))

def imresize(image, size):
    return np.array(Image.fromarray(image).resize(size))

def imsave(path, image):
    # Ensure the image is in uint8 format
    image = image.astype(np.uint8)
    Image.fromarray(image).save(path)

def padding(img, target_height, target_width, channels):
    original_height, original_width, _ = img.shape
    new_height, new_width = original_height, original_width

    # Compute the padding values
    pad_height = max(target_height - new_height, 0)
    pad_width = max(target_width - new_width, 0)

    # Pad the image
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    img_padded = np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_padded

def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0] * factor_scale_r))
        c = int(np.round(coord[1] * factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out

def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols),] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_image_pytorch(image):
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    return preprocess(image)

def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        original_map = imread(path)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, :, :, 0] = padded_map.astype(np.float32)
        ims[i, :, :, 0] /= 255.0

    return ims

def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        fix_map = scipy.io.loadmat(path)["I"]
        ims[i, :, :, 0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims

def postprocess_predictions(predictions, width, height):
    res = (predictions - predictions.min()) / (predictions.max() - predictions.min()) * 255
    res = res.astype('uint8')
    res = Image.fromarray(res).resize((width, height))
    return res

def pad_to_max_size(tensor_list):
    max_size = tuple(max(s) for s in zip(*[t.size() for t in tensor_list]))
    padded_tensors = []
    for t in tensor_list:
        padding = [(0, max_dim - curr_dim) for max_dim, curr_dim in zip(max_size, t.size())]
        padding.reverse()  # torch.nn.functional.pad expects padding in reverse order
        padding = [p for sublist in padding for p in sublist]  # flatten the list
        padded_tensors.append(F.pad(t, padding, "constant", 0))
    return torch.stack(padded_tensors)
