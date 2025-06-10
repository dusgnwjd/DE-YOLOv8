from ultralytics import YOLO
from mAP import calculate_map
import os

from align_face import align_face
import cv2
import os

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


def run_inference():
    preprocess_align_images("data/test/images")
    print("🚀 Inference 시작 중...")
    model = YOLO('runs/detect/train7/weights/best.pt')  # 학습된 모델 경로
    results = model.predict(
        source='data/test/images',     # 테스트 이미지 폴더
        imgsz=416,
        conf=0.25,
        save=True,
        save_txt=True,                 # 예측 결과 저장 (.txt)
        project='runs/detect/predict',
        name='test',
        exist_ok=True
    )
    print("✅ Inference 완료! 결과는 runs/detect/predict/test/ 에 저장됨.\n")

def evaluate_map():
    print("📊 mAP 평가 시작...")
    gt_dir = 'data/test/labels'
    pred_dir = 'runs/detect/predict/test/labels'

    # 라벨 디렉토리 존재 확인
    if not os.path.exists(pred_dir) or len(os.listdir(pred_dir)) == 0:
        print("❌ 예측 라벨이 없습니다. 먼저 inference를 수행하거나 save_txt=True 옵션을 확인하세요.")
        return

    map50, ap_per_class = calculate_map(gt_dir, pred_dir, iou_threshold=0.5)
    print(f"\n📌 mAP@0.5: {map50:.4f}")
    for i, ap in enumerate(ap_per_class):
        print(f"Class {i} AP: {ap:.4f}")
    print("✅ 평가 완료!")

if __name__ == '__main__':
    run_inference()
    evaluate_map()
