import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as V2
from torchvision.transforms import v2 
import cv2
import sys
import natsort
from torchvision import datasets as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import albumentations as A
import matplotlib.pyplot as plt

def tensor_to_cv2_image(tensor_image):
    """
    PyTorch tensor를 OpenCV 이미지로 변환
    Args:
        tensor_image: (C, H, W) 형태의 tensor 또는 (B, C, H, W) 배치
    Returns:
        OpenCV 이미지 (H, W, C) BGR 형태
    """
    # 배치에서 첫 번째 이미지 선택 (필요시)
    if len(tensor_image.shape) == 4:  # (B, C, H, W)
        tensor_image = tensor_image[0]
    
    # GPU tensor라면 CPU로 이동
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    
    # (C, H, W) -> (H, W, C)로 변환
    image = tensor_image.permute(1, 2, 0).numpy()
    
    # 정규화 해제: [-1, 1] -> [0, 1]
    # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 역변환
    image = image * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    
    # [0, 1] -> [0, 255]
    image = (image * 255).astype(np.uint8)
    
    # RGB -> BGR (OpenCV 형태)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image_bgr

def visualize_batch(tensor_batch, labels=None):
    #plt.figure(figsize=(12, 8))
    batch_size = tensor_batch.shape[0]
    
    for i in range(min(batch_size, 4)):  
        cv2_image = tensor_to_cv2_image(tensor_batch[i])
        # plt.subplot(2, 2, i + 1)
        # plt.title(f"Label: {labels[i].item()}" if labels is not None else "Image")
        # plt.imshow(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

class Dataset(data.Dataset):

    def __init__(self, root , phase , input_shape):
        self.phase = phase
        self.input_shape = input_shape

        self.image_paths = []
        self.labels = []

    
        folder_names = os.listdir(root)
        try:
            numeric_folders = [f for f in folder_names if f.isdigit()]
            self.classes = sorted(numeric_folders, key=int)  
        except:
            self.classes = natsort.natsorted(folder_names)
        
        class_to_idx = {cls_name: int(cls_name) if cls_name.isdigit() else i 
                       for i, cls_name in enumerate(self.classes)}

        self.CLAHE_transform = A.CLAHE(p=0.3, clip_limit=2.0, tile_grid_size=(8, 8))

        for cls_name in tqdm(self.classes):
            class_idx = class_to_idx[cls_name]
            class_dir = os.path.join(root , cls_name)

            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir , img_name)

                self.image_paths.append(img_path)
                self.labels.append(class_idx)

        if self.phase == 'train':
            self.transforms = V2.Compose([
                V2.ToTensor(),
                V2.CenterCrop(size=(112,112)),
                V2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                V2.RandomHorizontalFlip(p=0.3),
                V2.RandomRotation(degrees=10),
            ])


    @property
    def get_classes(self):
        return len(self.classes)


    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = np.int32(self.labels[index])
        
        path_list = []
        path_list.append(img_path)

        try:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"이미지를 읽을 수 없습니다: {img_path}")

        except Exception as e:
            raise FileNotFoundError(f"IMAGE path error: {img_path}, error: {e}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.CLAHE_transform(image=image)['image']  

        image = Image.fromarray(image_rgb)
        transformed_image = self.transforms(image).type(torch.float32)


        return transformed_image, label  , img_path

    def __len__(self):
        return len(self.image_paths)

class FilteredDataset(data.Dataset):
    def __init__(self, original_dataset, classes_to_keep , partical_Margin_product = False):
        """partical_Margin_prodcut HEAD MLP 전체 클래스수 -> 사용할 클래스수"""
        self.original_dataset = original_dataset
        self.partical_Margin_product = partical_Margin_product
        self.classes_to_keep = classes_to_keep # 사용할 클래스수
        self.class_mapping = {old_label: new_label for new_label, old_label in enumerate(self.classes_to_keep)} # {기존클래스 : 새로운클래스명}

        original_labels = np.array(self.original_dataset.labels)
        mask = np.isin(original_labels, self.classes_to_keep) # mask생성 [0,5,10,3] [5,10] -> [F,T,T,F]
        self.indices = np.where(mask)[0]

        self.remapped_labels = np.array([self.class_mapping[label] for label in original_labels[mask]])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        if self.partical_Margin_product:
            original_index = self.indices[index]
            image, original_label, path = self.original_dataset[original_index]
            return image, original_label, path
        
        else:
            original_index = self.indices[index]
            image, _, path = self.original_dataset[original_index]
            new_label = self.remapped_labels[index]
            return image, new_label, path
        

    @property
    def get_classes(self):
        return len(self.classes_to_keep)

if __name__ == '__main__':
    print("🚀 Dataset 시각화 테스트 시작...")
    
    dataset = Dataset(root='/home/ubuntu/arcface-pytorch/pair',
                      phase='train',
                      input_shape=(1, 112, 112))

    print(f"📊 Dataset 정보:")
    print(f"   - 총 이미지 수: {len(dataset)}")
    print(f"   - 클래스 수: {dataset.get_classes}")
    print(f"   - 첫 10개 클래스: {dataset.classes[:10]}")
    
    trainloader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    for index, (transformed, label, image_path) in tqdm(enumerate(trainloader)):

        answer_label = list(map(int, map(lambda x: x.split('/')[-2], image_path)))
        answer_label_tensor = torch.tensor(answer_label, dtype=torch.int32)
        
        print(f"\n📋 Batch {index + 1} 정보:")
        print(f"   - Transformed Image Shape: {transformed.shape}")
        print(f"   - Labels: {label.tolist()}")
        print(f"   - Answer Labels: {answer_label_tensor.tolist()}")
        print(f"   - Image Paths: {['/'.join(path.split('/')[-2 : ]) for path in image_path]}")

        if torch.equal(label, answer_label_tensor):
            print("   ✅ 라벨이 일치합니다!")
        else:
            raise ValueError("❌ Labels do not match! label: {}, answer_label: {} img_path: {}".format(
                label, answer_label_tensor, image_path))

        if index < 3:
            print(f"\n🖼️ Batch {index + 1} 이미지 시각화 중...")
           
            visualize_batch(transformed, labels=label)

            print(f"   📊 Tensor 통계:")
            print(f"      - Min: {transformed.min().item():.4f}")
            print(f"      - Max: {transformed.max().item():.4f}")
            print(f"      - Mean: {transformed.mean().item():.4f}")
            print(f"      - Std: {transformed.std().item():.4f}")
        
