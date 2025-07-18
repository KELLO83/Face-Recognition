import os
import sys
import numpy as np
from collections import Counter
import natsort
from tqdm import tqdm


class SimpleDataset:
    def __init__(self, root):
        self.image_paths = []
        self.labels = []
        self.classes = natsort.natsorted(os.listdir(root))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print("데이터 로딩...")
        for cls_name in tqdm(self.classes):
            class_idx = class_to_idx[cls_name]
            class_dir = os.path.join(root, cls_name)
            
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)

def test_labeling(root_path='mini_arcface'):
    """레이블링이 정상적으로 되었는지 테스트하는 함수"""
    
    print(f"=== {root_path} 폴더 구조 및 레이블링 테스트 ===\n")
    
    # 폴더 구조 확인
    if not os.path.exists(root_path):
        print(f"❌ 폴더가 존재하지 않습니다: {root_path}")
        return
    
    folders = sorted(os.listdir(root_path))
    print("📁 폴더 구조:")
    for i, folder in enumerate(folders):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path):
            image_count = len([f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"  {i}: {folder}/ ({image_count}개 이미지)")
    print()
    
    # Dataset 객체 생성
    try:
        dataset = SimpleDataset(root=root_path)
        print(f"✅ Dataset 생성 성공!")
        print(f"   총 이미지 수: {len(dataset)}")
        print(f"   클래스 수: {len(dataset.classes)}")
        print(f"   클래스 목록: {dataset.classes}")
        print()
        
        # 레이블 분포 확인
        label_counts = Counter(dataset.labels)
        print("📊 레이블 분포:")
        for label, count in sorted(label_counts.items()):
            class_name = dataset.classes[label]
            print(f"   레이블 {label} (폴더 '{class_name}'): {count}개 이미지")
        print()
        
        # 샘플 이미지 경로와 레이블 확인
        print("🔍 샘플 이미지 경로와 레이블 확인:")
        for i in range(min(10, len(dataset))):  # 처음 10개만 확인
            img_path = dataset.image_paths[i]
            label = dataset.labels[i]
            folder_name = os.path.basename(os.path.dirname(img_path))
            class_name = dataset.classes[label]
            
            # 폴더명과 클래스명이 일치하는지 확인
            is_correct = folder_name == class_name
            status = "✅" if is_correct else "❌"
            
            print(f"   {status} {img_path}")
            print(f"      레이블: {label}, 클래스: '{class_name}', 폴더: '{folder_name}'")
            
            if not is_correct:
                print(f"      ⚠️  경고: 폴더명과 클래스명이 일치하지 않습니다!")
            print()
        
        # 각 폴더별로 레이블 일관성 확인
        print("🔎 폴더별 레이블 일관성 검사:")
        folder_labels = {}
        for img_path, label in zip(dataset.image_paths, dataset.labels):
            folder_name = os.path.basename(os.path.dirname(img_path))
            if folder_name not in folder_labels:
                folder_labels[folder_name] = set()
            folder_labels[folder_name].add(label)
        
        all_consistent = True
        for folder_name, labels_set in folder_labels.items():
            if len(labels_set) == 1:
                label = list(labels_set)[0]
                print(f"   ✅ 폴더 '{folder_name}': 모든 이미지가 레이블 {label}로 일관됨")
            else:
                print(f"   ❌ 폴더 '{folder_name}': 여러 레이블 발견 {labels_set}")
                all_consistent = False
        
        print()
        if all_consistent:
            print("🎉 모든 폴더의 레이블링이 일관되게 정상 작동합니다!")
        else:
            print("⚠️  일부 폴더에서 레이블링 문제가 발견되었습니다.")
            
    except Exception as e:
        print(f"❌ Dataset 생성 중 오류 발생: {e}")

if __name__ == '__main__':
    test_labeling('dataset/ms1m-arcface')
