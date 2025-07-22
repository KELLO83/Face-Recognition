import os
import itertools
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import logging
import traceback
import pickle
import argparse
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, cpu_count
from insightface.app.face_analysis import FaceAnalysis
import cv2

# --- 로깅 및 경로 설정 ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
LOG_FILE = os.path.join(script_dir, "insface_log.txt")

logging.basicConfig(
    filename=LOG_FILE, level=logging.WARNING,
    format='%(a평가에sctime)s - %(levelname)s - %(message)s', filemode='w'
)

# --- 병렬 처리를 위한 전역 모델 및 초기화 함수 ---
worker_app = None

def init_worker(model_name, root_dir):
    """각 워커 프로세스를 위한 초기화 함수. 모델을 한 번만 로드합니다."""
    global worker_app
    # print(f"Initializing worker {os.getpid()} with model {model_name}...")
    worker_app = FaceAnalysis(name=model_name,
                              providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                              root=root_dir)
    worker_app.prepare(ctx_id=0, det_size=(640, 640))

def represent_image_insface(img_path):
    """InsightFace 모델을 사용하여 이미지에서 임베딩을 추출합니다."""
    global worker_app
    try:
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"이미지를 읽을 수 없습니다: {img_path}")
            return img_path, None
        
        faces = worker_app.get(img)
        if not faces:
            logging.warning(f"얼굴을 찾을 수 없습니다: {img_path}")
            return img_path, None
            
        return img_path, faces[0]['embedding']
    except Exception as e:
        logging.warning(f"임베딩 추출 중 오류 발생: {img_path}. 오류: {e}")
        return img_path, None

def get_all_embeddings(identity_map, model_name, dataset_name, use_cache=True):
    """임베딩을 추출하거나 캐시에서 로드 (병렬 처리 기능 포함)"""
    cache_file = os.path.join(script_dir, f"embeddings_cache_{dataset_name}_{model_name}_insightface.pkl")
    
    if use_cache and os.path.exists(cache_file):
        print(f"\n캐시 파일 '{cache_file}'에서 임베딩을 로드합니다...")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
        print("임베딩 로드 완료.")
        return embeddings

    all_images = sorted(list(set(itertools.chain.from_iterable(identity_map.values()))))
    print(f"\n총 {len(all_images)}개의 이미지에 대해 InsightFace 임베딩을 새로 추출합니다 (병렬 처리 사용)...")
    
    embeddings = {}
    
    # 시스템 부하를 고려하여 CPU 코어 수 - 1 만큼의 프로세스 사용
    # initializer를 사용하여 각 프로세스 시작 시 모델을 한 번만 로드
    with Pool(processes=max(1, cpu_count() - 1), initializer=init_worker, initargs=(model_name, ".")) as pool:
        results = list(tqdm(pool.imap(represent_image_insface, all_images), total=len(all_images), desc="임베딩 추출"))

    for img_path, embedding in results:
        embeddings[img_path] = embedding

    if use_cache:
        print(f"\n추출된 임베딩을 캐시 파일 '{cache_file}'에 저장합니다...")
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print("캐시 저장 완료.")
    return embeddings

def collect_scores_from_embeddings(pairs, embeddings, is_positive):
    """임베딩으로 거리를 계산합니다."""
    distances, labels = [], []
    label = 1 if is_positive else 0
    desc = "동일 인물 쌍 계산" if is_positive else "다른 인물 쌍 계산"
    for img1_path, img2_path in tqdm(pairs, desc=desc):
        emb1, emb2 = embeddings.get(img1_path), embeddings.get(img2_path)
        if emb1 is not None and emb2 is not None:
            distances.append(cosine(emb1, emb2))
            labels.append(label)
    return distances, labels

def save_results_to_excel(excel_path, model_name, roc_auc, eer, tar_at_far_results, target_fars, metrics):
    """결과를 Excel 파일에 저장합니다."""
    new_data = {
        "model_name": [model_name], "detector_backend": ["insightface"],
        "roc_auc": [f"{roc_auc:.4f}"], "eer": [f"{eer:.4f}"],
        "accuracy": [f"{metrics['accuracy']:.4f}"], "recall": [f"{metrics['recall']:.4f}"],
        "f1_score": [f"{metrics['f1_score']:.4f}"], "tp": [metrics['tp']],
        "tn": [metrics['tn']], "fp": [metrics['fp']], "fn": [metrics['fn']]
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
    print(f"\n평가 결과가 '{excel_path}' 파일에 저장되었습니다.")

def plot_roc_curve(fpr, tpr, roc_auc, model_name, excel_path):
    """ROC 커브를 그리고 파일로 저장합니다."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title(f'ROC Curve for {model_name} (InsightFace)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_filename = os.path.splitext(excel_path)[0] + f"_{model_name}_roc_curve.png"
    plt.savefig(plot_filename)
    print(f"ROC 커브 그래프가 '{plot_filename}' 파일로 저장되었습니다.")

def main(args):
    # --- 1단계: 데이터셋 스캔 ---
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {args.data_path}")

    identity_map = {}
    for person_folder in os.listdir(args.data_path):
        person_path = os.path.join(args.data_path, person_folder)
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > 1:
                identity_map[person_folder] = images
    
    if not identity_map:
        raise ValueError("데이터셋에서 2개 이상의 이미지를 가진 인물을 찾지 못했습니다.")
    print(f"총 {len(identity_map)}명의 인물, {sum(len(v) for v in identity_map.values())}개의 이미지를 찾았습니다.")

    # --- 2단계: 평가 쌍 생성 ---
    print("\n평가에 사용할 동일 인물/다른 인물 쌍을 생성합니다...")
    positive_pairs = [p for imgs in identity_map.values() for p in itertools.combinations(imgs, 2)]
    num_positive_pairs = len(positive_pairs)
    
    identities = list(identity_map.keys())
    negative_pairs_set = set()
    if len(identities) > 1:
        # 더 효율적인 음성 쌍 생성을 위해 노력
        max_attempts = num_positive_pairs * 5
        attempts = 0
        while len(negative_pairs_set) < num_positive_pairs and attempts < max_attempts:
            id1, id2 = random.sample(identities, 2)
            pair = (random.choice(identity_map[id1]), random.choice(identity_map[id2]))
            sorted_pair = tuple(sorted(pair))
            negative_pairs_set.add(sorted_pair)
            attempts += 1

    negative_pairs = list(negative_pairs_set)
    print(f"- 동일 인물 쌍: {len(positive_pairs)}개, 다른 인물 쌍: {len(negative_pairs)}개")

    # --- 3단계: 임베딩 추출 또는 캐시 로드 ---
    dataset_name = os.path.basename(os.path.normpath(args.data_path))
    embeddings = get_all_embeddings(identity_map, args.model_name, dataset_name, use_cache=not args.no_cache)

    # --- 4단계: 점수 수집 ---
    print("\n미리 계산된 임베딩으로 거리를 계산합니다...")
    pos_distances, pos_labels = collect_scores_from_embeddings(positive_pairs, embeddings, is_positive=True)
    neg_distances, neg_labels = collect_scores_from_embeddings(negative_pairs, embeddings, is_positive=False)
    
    if not pos_distances and not neg_distances:
        msg = "유효한 임베딩을 가진 쌍이 없어 평가를 진행할 수 없습니다. 로그 파일을 확인하세요."
        print(msg)
        logging.error(msg)
        return

    distances = np.array(pos_distances + neg_distances)
    labels = np.array(pos_labels + neg_labels)

    # --- 5단계: 최종 평가 및 결과 저장 ---
    print("\n--- 최종 평가 결과 ---")
    if labels.size > 0:
        scores = -distances  # 거리는 낮을수록 유사하므로, 점수는 부호를 반전
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        frr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fpr - frr))
        eer = fpr[eer_index]
        eer_threshold = -thresholds[eer_index] # 점수(유사도)가 아닌 거리 기준 임계값

        tar_at_far_results = {far: np.interp(far, fpr, tpr) for far in args.target_fars}
        
        predictions = (distances < eer_threshold).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else: # 모든 쌍이 동일인물이거나 다른인물일 경우 예외 처리
            tn, fp, fn, tp = 0, 0, 0, 0
            if len(cm) == 1:
                if all(labels == 1): tp = cm[0][0]
                else: tn = cm[0][0]

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # TAR
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics = {"accuracy": accuracy, "recall": recall, "f1_score": f1_score, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

        print(f"사용된 모델: {args.model_name} (InsightFace), 전체 평가 쌍: {len(labels)} 개")
        print(f"[주요 성능] ROC-AUC: {roc_auc:.4f}, EER: {eer:.4f} (거리 임계값: {eer_threshold:.4f})")
        print(f"[상세 지표] Accuracy: {metrics['accuracy']:.4f}, Recall (TAR): {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        for far, tar in tar_at_far_results.items():
            print(f"  - TAR @ FAR {far*100:g}%: {tar:.4f}")

        excel_path = os.path.join(script_dir, args.excel_path)
        save_results_to_excel(excel_path, args.model_name, roc_auc, eer, tar_at_far_results, args.target_fars, metrics)
        
        if args.plot_roc:
            plot_roc_curve(fpr, tpr, roc_auc, args.model_name, excel_path)
    else:
        msg = "평가를 위한 유효한 점수를 수집하지 못했습니다."
        print(msg)
        logging.error(msg)

if __name__ == "__main__":
    # CUDA와 multiprocessing 충돌 방지를 위해 'spawn' 시작 방식 사용
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="InsightFace Face Recognition Evaluation Script")
    parser.add_argument("--data_path", type=str, default="/home/ubuntu/Face-Recognition/backup/01.Recognition", help="평가할 데이터셋의 루트 폴더")
    parser.add_argument("--model_name", type=str, default="antelopev2", help="사용할 InsightFace 모델 (e.g., antelopev2, buffalo_l)")
    parser.add_argument("--excel_path", type=str, default="insface_evaluation_results.xlsx", help="결과를 저장할 Excel 파일 이름")
    parser.add_argument("--target_fars", nargs='+', type=float, default=[0.01, 0.001, 0.0001], help="TAR을 계산할 FAR 목표값들")
    parser.add_argument("--no-cache", action="store_true", help="이 플래그를 사용하면 기존 임베딩 캐시를 무시하고 새로 추출합니다.")
    parser.add_argument("--plot-roc", action="store_true", help="이 플래그를 사용하면 ROC 커브 그래프를 파일로 저장합니다.", default=True)
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        error_message = traceback.format_exc()
        logging.error(f"스크립트 실행 중 처리되지 않은 예외 발생:\n{error_message}")
        print(f"\n치명적인 오류가 발생했습니다. '{LOG_FILE}' 파일에 상세 내역이 기록되었습니다.")
