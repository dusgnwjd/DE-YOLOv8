
import cv2
import dlib
import numpy as np

# dlib predictor 로드 (초기화 시 1회만 호출 필요)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

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
    
def align_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return img  # 얼굴 없으면 원본 반환

    face = faces[0]
    landmarks = predictor(gray, face)

    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned
