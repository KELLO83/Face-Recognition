import os
import itertools
import random
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix

# --- 설정 ---
# 1. 평가할 데이터셋의 루트 폴더
DATASET_PATH = "/home/ubuntu/insightface/python-package/insightface/dataset/01.Recognition"

# 2. 평가에 사용할 모델 설정
MODEL_NAME = "ArcFace" # VGG-Face, Facenet, Facenet512, ArcFace, SFace 등
DETECTOR_BACKEND = "retinaface"

# 3. TAR@FAR 계산을 위한 목표 FAR 값들
TARGET_FARS = [0.01, 0.001, 0.0001] # 1%, 0.1%, 0.01%

def save_results_to_excel(model_name, detector_backend, roc_auc, eer, tar_at_far_results, target_fars, metrics):
    """
    평가 결과를 Excel 파일에 기록하고, 파일이 존재하면 새로운 행을 추가합니다.
    """
    excel_path = "evaluation_results_FAST.xlsx"
    
    # 새로운 결과 데이터 생성
    new_data = {
        "model_name": [model_name],
        "detector_backend": [detector_backend],
        "roc_auc": [f"{roc_auc:.4f}"],
        "eer": [f"{eer:.4f}"],
        "accuracy": [f"{metrics['accuracy']:.4f}"],
        "recall": [f"{metrics['recall']:.4f}"],
        "f1_score": [f"{metrics['f1_score']:.4f}"],
        "tp": [metrics['tp']],
        "tn": [metrics['tn']],
        "fp": [metrics['fp']],
        "fn": [metrics['fn']]
    }
    for far in target_fars:
        new_data[f"tar_at_far_{far*100:g}%"] = [f"{tar_at_far_results.get(far, 0):.4f}"]
    
    new_df = pd.DataFrame(new_data)

    # Excel 파일이 존재하면 기존 데이터를 읽어오고, 없으면 새로 만듭니다.
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    # 새로운 데이터를 추가하고 파일을 저장합니다.
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_excel(excel_path, index=False)
    print(f"\n평가 결과가 '{excel_path}' 파일에 저장되었습니다.")

# --- 1단계: 데이터셋 스캔 및 이미지 목록 생성 ---
print("데이터셋을 스캔하여 인물별 이미지 목록을 생성합니다...")
identity_map = {}
for person_folder in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_folder)
    if os.path.isdir(person_path):
        images = [os.path.join(person_path, img) for img in os.listdir(person_path)]
        if len(images) > 1:
            identity_map[person_folder] = images

print(f"총 {len(identity_map)}명의 인물을 찾았습니다.")

# --- 2단계: 평가용 이미지 쌍 생성 ---
print("평가에 사용할 동일 인물/다른 인물 쌍을 생성합니다...")
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

# --- 3단계: 모델 빌드 및 GPU 설정 ---
# 스크립트 시작 시 한 번만 모델을 빌드하여 GPU 메모리에 올립니다.
# 이를 통해 이후의 모든 verify 호출이 GPU를 사용하여 빠르게 처리됩니다.
print(f"모델({MODEL_NAME})을 빌드하고 GPU에 로드합니다...")
try:
    DeepFace.build_model(MODEL_NAME)
    print("모델이 성공적으로 빌드되었습니다.")
except Exception as e:
    print(f"모델 빌드 중 오류 발생: {e}")
    print("CPU로 계속 진행합니다.")

# --- 4단계: 모델 평가 실행 및 점수 수집 ---
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

# --- 4단계: 고급 성능 지표 계산 및 출력 ---
print("\n--- 최종 평가 결과 ---")

if not labels:
    print("평가를 위한 유효한 점수를 수집하지 못했습니다. 데이터셋을 확인하세요.")
    # `distance`는 낮을수록 좋으므로, roc_curve를 위해 (1-distance) 또는 (-distance)로 변환
    # 여기서는 점수처럼 활용하기 위해 음수로 변환 (높을수록 좋음)
    scores = -np.array(distances)
    labels = np.array(labels)

    # ROC 커브 계산
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # 1. ROC-AUC 계산
    roc_auc = auc(fpr, tpr)

    # 2. EER (Equal Error Rate) 계산
    frr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - frr))
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]

    # 3. 특정 FAR 값에서의 TAR 계산
    tar_at_far_results = {}
    for target_far in TARGET_FARS:
        # np.interp: fpr 배열에서 target_far에 가장 가까운 지점의 tpr 값을 보간하여 찾아줌
        tar_at_far = np.interp(target_far, fpr, tpr)
        tar_at_far_results[target_far] = tar_at_far

    # 4. EER 임계값 기준 추가 성능 지표 계산
    predictions = scores > eer_threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

    # 결과 출력
    print(f"사용된 모델: {MODEL_NAME}")
    print(f"전체 평가 쌍: {len(labels)} 개")
    print("\n[주요 성능 지표]")
    print(f"- ROC-AUC: {roc_auc:.4f} (1에 가까울수록 모델의 전반적인 성능이 좋음)")
    print(f"- EER (Equal Error Rate): {eer:.4f} (값이 낮을수록 좋음)")
    print(f"  (EER은 FAR과 FRR이 같아지는 지점, 보안과 편의의 균형점)")
    print(f"  (EER 임계값: {-eer_threshold:.4f}) (이 distance 값에서 EER 발생)")
    
    print("\n[EER 임계값 기준 상세 지표]")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Recall (TPR): {recall:.4f}")
    print(f"- F1-Score: {f1_score:.4f}")
    print(f"- TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    print("\n- TAR @ FAR (특정 오인식률에서의 본인 인증 성공률)")
    for far, tar in tar_at_far_results.items():
        print(f"  - FAR = {far*100:g}% 일 때, TAR = {tar:.4f}")

    # --- 5단계: Excel 파일에 결과 저장 ---
    save_results_to_excel(
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        roc_auc=roc_auc,
        eer=eer,
        tar_at_far_results=tar_at_far_results,
        target_fars=TARGET_FARS,
        metrics=metrics
    )

    # (선택) ROC 커브를 시각화하여 저장하고 싶다면 아래 주석을 해제하세요.
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate (FAR)')
    # plt.ylabel('True Positive Rate (TAR)')
    # plt.title(f'ROC Curve for {MODEL_NAME}')
    # plt.legend(loc="lower right")
    # plt.savefig(f'{MODEL_NAME}_roc_curve.png')
    # print(f"\nROC 커브 그래프가 '{MODEL_NAME}_roc_curve.png' 파일로 저장되었습니다.")