import os
import cv2


def read_float_with_comma(num):
    return float(num.replace(",", "."))


def extract_frame(videos_dir, video_name, size, save_folder, video_length_min, frames_per_second=5):
    """
    Extract frames from a video file, sampling multiple frames per second.

    Parameters:
    - videos_dir: str, path to the directory containing video files.
    - video_name: str, name of the video file (without extension).
    - save_folder: str, path to the directory where extracted frames will be saved.
    - size: int, target size for resizing frames.
    - video_length_min: int, minimum number of frames to extract.
    - frames_per_second: int, how many frames to extract per second.
    """
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]

    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print(f"Error: Couldn't open video file {filename}")
        return

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the height of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of frames

    # Resize based on video orientation
    if video_height > video_width:
        video_width_resize = size
        video_height_resize = int(video_width_resize / video_width * video_height)
    else:
        video_height_resize = size
        video_width_resize = int(video_height_resize / video_height * video_width)

    dim = (video_width_resize, video_height_resize)

    video_read_index = 0
    frame_idx = 0
    frames_to_sample = frames_per_second * video_frame_rate

    # Calculate uniform sampling intervals
    sampling_interval = max(1, video_frame_rate // frames_per_second)

    for i in range(video_length):
        has_frames, frame = cap.read()
        if has_frames:
            # Sample frames at the calculated interval
            if (video_read_index < video_length) and (frame_idx % sampling_interval == 0):
                read_frame = cv2.resize(frame, dim)
                exist_folder(os.path.join(save_folder, video_name_str))
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                         '{:03d}'.format(video_read_index) + '.png'), read_frame)
                video_read_index += 1
            frame_idx += 1

    # If the number of extracted frames is less than the video_length_min, copy the last frame
    if video_read_index < video_length_min:
        for i in range(video_read_index, video_length_min):
            exist_folder(os.path.join(save_folder, video_name_str))
            cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                     '{:03d}'.format(i) + '.png'), read_frame)

    return


def exist_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return