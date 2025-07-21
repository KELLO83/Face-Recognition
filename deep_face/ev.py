import os
import itertools
import random
import numpy as np
from deepface import DeepFace
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc


DATASET_PATH = "/home/ubuntu/insightface/python-package/insightface/dataset/01.Recognition"


MODEL_NAME = "ArcFace" 
DETECTOR_BACKEND = "retinaface"
TARGET_FARS = [0.01, 0.001, 0.0001] # 1%, 0.1%, 0.01%


identity_map = {}
for person_folder in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_folder)
    if os.path.isdir(person_path):
        images = [os.path.join(person_path, img) for img in os.listdir(person_path)]
        if len(images) > 1:
            identity_map[person_folder] = images

print(f"총 {len(identity_map)}명의 인물을 찾았습니다.")


positive_pairs = []
for person, images in tqdm(identity_map.items()):
    for pair in itertools.combinations(images, 2):
        positive_pairs.append(pair)

negative_pairs = []
num_positive_pairs = len(positive_pairs)
identities = list(identity_map.keys())

if len(identities) > 1:
    while len(negative_pairs) < num_positive_pairs:
        id1, id2 = random.sample(identities, 2)
        img1 = random.choice(identity_map[id1])
        img2 = random.choice(identity_map[id2])
        pair = (img1, img2)
        if pair not in negative_pairs:
            negative_pairs.append(pair)

print(f"- 동일 인물 쌍 (Positive Pairs): {len(positive_pairs)}개")
print(f"- 다른 인물 쌍 (Negative Pairs): {len(negative_pairs)}개")


print(f"모델({MODEL_NAME})을 빌드하고 GPU에 로드합니다...")
try:
    DeepFace.build_model(MODEL_NAME)
    print("모델이 성공적으로 빌드되었습니다.")
except Exception as e:
    print(f"모델 빌드 중 오류 발생: {e}")
    print("CPU로 계속 진행합니다.")


labels = []
distances = []

def collect_scores(pairs, is_positive):
    label = 1 if is_positive else 0
    desc = "동일 인물 쌍 평가 중" if is_positive else "다른 인물 쌍 평가 중"
    
    for img1_path, img2_path in tqdm(pairs, desc=desc):
        try:
            result = DeepFace.verify(
                img1_path=img1_path, 
                img2_path=img2_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )
            # `distance`는 거리를 의미하므로 낮을수록 유사함
            distances.append(result['distance'])
            labels.append(label)
        except Exception:
            pass # 오류 발생 시 해당 쌍은 건너뜀

collect_scores(positive_pairs, is_positive=True)
collect_scores(negative_pairs, is_positive=False)

print("\n--- 최종 평가 결과 ---")

if not labels:
    print("평가를 위한 유효한 점수를 수집하지 못했습니다. 데이터셋을 확인하세요.")
else:
    scores = -np.array(distances)
    labels = np.array(labels)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)

    frr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - frr))
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]

    tar_at_far_results = {}
    for target_far in TARGET_FARS:
        tar_at_far = np.interp(target_far, fpr, tpr)
        tar_at_far_results[target_far] = tar_at_far

    # 결과 출력
    print(f"사용된 모델: {MODEL_NAME}")
    print(f"전체 평가 쌍: {len(labels)} 개")
    print("\n[주요 성능 지표]")
    print(f"- ROC-AUC: {roc_auc:.4f} (1에 가까울수록 모델의 전반적인 성능이 좋음)")
    print(f"- EER (Equal Error Rate): {eer:.4f} (값이 낮을수록 좋음)")
    print(f"  (EER은 FAR과 FRR이 같아지는 지점, 보안과 편의의 균형점)")
    print(f"  (EER 임계값: {-eer_threshold:.4f}) (이 distance 값에서 EER 발생)")
    
    print("\n- TAR @ FAR (특정 오인식률에서의 본인 인증 성공률)")
    for far, tar in tar_at_far_results.items():
        print(f"  - FAR = {far*100:g}% 일 때, TAR = {tar:.4f}")

    #(선택) ROC 커브를 시각화하여 저장하고 싶다면 아래 주석을 해제하세요.
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title(f'ROC Curve for {MODEL_NAME}')
    plt.legend(loc="lower right")
    plt.savefig(f'{MODEL_NAME}_roc_curve.png')
    print(f"\nROC 커브 그래프가 '{MODEL_NAME}_roc_curve.png' 파일로 저장되었습니다.")