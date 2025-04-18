import os
import cv2
import rarfile
import patoolib
import tempfile
import shutil

def read_UCF11(data_dir, num_frames):
    """
    Reads video data, applies transformations, and extracts features using ResNet50. 
    This function is used for UCF11 dataset.
    """
    samples= [] # List to store the features and labels
    # Loop over the videos in the dataset folder
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        print(label_dir)
        for sub_dir in os.listdir(label_dir):
            if sub_dir == 'Annotation':
                continue
            video_dir = os.path.join(label_dir, sub_dir)
            for video_file in os.listdir(video_dir):
                video_path = os.path.join(video_dir, video_file)
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                frames = []
                while True:
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        if frame_count == num_frames:
                            break
                    else:
                        break
                cap.release()
                if len(frames) == num_frames:
                    samples.append((frames, label))
    return samples

def read_UCF50(data_dir, num_frames):
    """
    Reads video data, applies transformations, and extracts features using ResNet50. 
    This function is used for UCF50 dataset.
    """
    samples= [] # List to store the features and labels
    # Loop over the videos in the dataset folder
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        print(label_dir)
        for video_file in os.listdir(label_dir):
            video_path = os.path.join(label_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    if frame_count == num_frames:
                        break
                else:
                    break
            cap.release()
            if len(frames) == num_frames:
                samples.append((frames, label))
    return samples

def read_HMDB51(data_dir, num_frames):
    """
    Reads video data, applies transformations, and extracts features using ResNet50. 
    This function is used for HMDB51 dataset.
    """
    samples= [] # List to store the features and labels
    # Loop over the videos in the dataset folder
    for rar_file in os.listdir(data_dir):
        if rar_file.endswith('.rar'):
            rar_path = os.path.join(data_dir, rar_file)
            print(rar_file)

            label = os.path.splitext(rar_file)[0]
            extract_dir = os.path.join(data_dir, label)

            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir, exist_ok=True)

                # Create a temp dir for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    patoolib.extract_archive(rar_path, outdir=temp_dir)

                    # Move all .avi files from temp_dir (including subfolders) to extract_dir
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('.avi'):
                                src = os.path.join(root, file)
                                dst = os.path.join(extract_dir, file)
                                shutil.move(src, dst)

            # Process the extracted files
            for video_file in os.listdir(extract_dir):
                video_path = os.path.join(extract_dir, video_file)
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                frames = []
                while True:
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        if frame_count == num_frames:
                            break
                    else:
                        break
                cap.release()
                if len(frames) == num_frames:
                    samples.append((frames, label))
    return samples
