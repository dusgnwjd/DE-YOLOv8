from ultralytics import YOLO
import torch.nn as nn
import torchvision.models as models
import torch
from dynamic_erasing import DynamicErasing
from ultralytics import YOLO
import random
import numpy as np
from align_face import align_face
import cv2
import os


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN의 비결정성 방지
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def preprocess_align_images(image_dir):
    print(f"📐 얼굴 정렬 중: {image_dir}")
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        aligned = align_face(img)
        cv2.imwrite(img_path, aligned)
    print(f"✅ 얼굴 정렬 완료")

def main():
    preprocess_align_images("data/train/images")
    preprocess_align_images("data/valid/images")
    preprocess_align_images("data/test/images")
    yolo = YOLO('yolov8s.yaml')  # 구조만 불러오기
    seed_everything(42)
    model = yolo.model
    # backbone = model.model[0], neck = model.model[1], head = model.model[2]
    # 🔁 backbone + dynamic erase + neck + head
    backbone = model.model[0]
    neck = model.model[1]
    head = model.model[2]

    # 🔧 new model with dynamic erasing after backbone
    new_model = nn.Sequential(
        backbone,
        DynamicErasing(erase_prob=0.5, erase_ratio=0.2),
        neck,
        head
    )

    yolo.model.model = new_model
    yolo.model.nc = 1
    yolo.model.names = ['face']

    yolo.train(
        data='data.yaml',
        epochs=70,
        imgsz=416,
        batch=64,
        seed=42,
        save_conf=True
    )

if __name__ == '__main__':
    import multiprocessing
    seed_everything(42)
    multiprocessing.freeze_support()
    main()
