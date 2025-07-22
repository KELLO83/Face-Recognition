import os
import itertools
import random
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import logging
import traceback


try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
LOG_FILE = os.path.join(script_dir, "log.txt")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  
)

def main():

    DATASET_PATH = "/home/ubuntu/Face-Recognition/backup/01.Recognition"

    MODEL_NAME = "ArcFace"  
    DETECTOR_BACKEND = "retinaface"

    TARGET_FARS = [0.01, 0.001, 0.0001]  # 1%, 0.1%, 0.01%

    def save_results_to_excel(model_name, detector_backend, roc_auc, eer, tar_at_far_results, target_fars, metrics):
        excel_path = os.path.join(script_dir, "evaluation_results_FAST.xlsx")
        
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

        try:
            df = pd.read_excel(excel_path)
        except FileNotFoundError:
            df = pd.DataFrame()

        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_df.to_excel(excel_path, index=False)
        try:
            with open('evaluation_results_FAST.txt', 'w') as f:
                f.write(updated_df.to_string(index=False))
        except Exception as e:
            logging.error(f"평가 결과를 텍스트 파일로 저장하는 중 오류 발생: {e}")
        print(f"\n평가 결과가 '{excel_path}' 파일에 저장되었습니다.")

    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {DATASET_PATH}")

    identity_map = {}
    for person_folder in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_folder)
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, img) for img in os.listdir(person_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > 1:
                identity_map[person_folder] = images
    
    if not identity_map:
        raise ValueError("데이터셋에서 2개 이상의 이미지를 가진 인물을 찾지 못했습니다. 폴더 구조를 확인하세요.")

    print(f"총 {len(identity_map)}명의 인물을 찾았습니다.")

    # --- 2단계: 평가용 이미지 쌍 생성 ---
    print("평가에 사용할 동일 인물/다른 인물 쌍을 생성합니다...")
    positive_pairs = []
    for person, images in tqdm(identity_map.items(), desc="동일 인물 쌍 생성"):
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

            if pair not in negative_pairs and (img2, img1) not in negative_pairs:
                negative_pairs.append(pair)

    print(f"- 동일 인물 쌍 (Positive Pairs): {len(positive_pairs)}개")
    print(f"- 다른 인물 쌍 (Negative Pairs): {len(negative_pairs)}개")

    print(f"모델({MODEL_NAME})을 빌드하고 GPU에 로드합니다...")
    DeepFace.build_model(MODEL_NAME)
    print("모델이 성공적으로 빌드되었습니다.")

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
                distances.append(result['distance'])
                labels.append(label)
            except Exception as e:
                logging.warning(f"이미지 쌍 평가 오류: ({img1_path}, {img2_path}). 건너뜁니다. 오류: {e}")
                pass

    collect_scores(positive_pairs, is_positive=True)
    collect_scores(negative_pairs, is_positive=False)

    print("\n--- 최종 평가 결과 ---")

    if labels and len(labels) == len(distances):
        scores = -np.array(distances)
        labels = np.array(labels)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        frr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fpr - frr))
        eer = fpr[eer_index]
        eer_threshold = thresholds[eer_index]

        tar_at_far_results = {far: np.interp(far, fpr, tpr) for far in TARGET_FARS}

        predictions = scores > eer_threshold

        cm = confusion_matrix(labels, predictions)
        if cm.size == 1: 
            if labels[0] == 0: 
                tn, fp, fn, tp = cm[0][0], 0, 0, 0

            else: 
                tn, fp, fn, tp = 0, 0, 0, cm[0][0]
        else:
            tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {"accuracy": accuracy, "recall": recall, "f1_score": f1_score, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

        print(f"사용된 모델: {MODEL_NAME}")
        print(f"전체 평가 쌍: {len(labels)} 개")
        print("\n[주요 성능 지표]")
        print(f"- ROC-AUC: {roc_auc:.4f}")
        print(f"- EER: {eer:.4f} (임계값: {-eer_threshold:.4f})")
        
        print("\n[EER 임계값 기준 상세 지표]")
        print(f"- Accuracy: {accuracy:.4f}, Recall (TPR): {recall:.4f}, F1-Score: {f1_score:.4f}")
        print(f"- TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        print("\n- TAR @ FAR")
        for far, tar in tar_at_far_results.items():
            print(f"  - FAR = {far*100:g}% 일 때, TAR = {tar:.4f}")

        save_results_to_excel(MODEL_NAME, DETECTOR_BACKEND, roc_auc, eer, tar_at_far_results, TARGET_FARS, metrics)
    else:
        msg = "평가를 위한 유효한 점수를 수집하지 못했습니다. 데이터셋, 모델 설정 또는 개별 이미지 파일들을 확인하세요."
        print(msg)
        logging.error(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_message = traceback.format_exc()
        logging.error(f"스크립트 실행 중 처리되지 않은 예외 발생:\n{error_message}")
        print(f"\n치명적인 오류가 발생했습니다. '{LOG_FILE}' 파일에 상세 내역이 기록되었습니다.")
