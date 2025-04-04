import os
import cv2

def readData(data_dir, num_frames):
    """
    Reads video data, applies transformations, and extracts features using ResNet50. 
    This function is used for UCF11_updated_mpg dataset.
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