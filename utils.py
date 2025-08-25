import cv2
import numpy as np
from PIL import Image

def extract_key_frames(video_path, num_frames=5):
    """
    Trích xuất các khung hình gần cuối có chất lượng cao
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)
    frames = []

    for i in range(total_frames - num_frames * step, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def check_quality(frame):
    """
    Kiểm tra chất lượng ảnh dựa trên độ nét
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def generate_prompt(frame):
    """
    Sinh prompt mô tả ảnh (có thể tích hợp model AI thật)
    """
    return f"(Một khung hình với độ phân giải {frame.shape[1]}x{frame.shape[0]}, màu sắc tự nhiên, chi tiết cao)"
