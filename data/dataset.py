import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as V2
import cv2
import sys
import natsort
from torchvision import datasets as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy


class Dataset(data.Dataset):

    def __init__(self, root , phase , input_shape):
        self.phase = phase
        self.input_shape = input_shape

        self.image_paths = []
        self.labels = []

        self.classes = natsort.natsorted(os.listdir(root)) # 폴더 갯수만큼 클래스 갯수를만듬
        class_to_idx = {cls_name : i for i , cls_name in enumerate(self.classes)} # 폴더명과 레이블 idx매칭 ex) 10번폴더는 2번..

        for cls_name in tqdm(self.classes):
            class_idx = class_to_idx[cls_name] # 10번폴더는 2번
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
        image = Image.fromarray(image_rgb)

        transformed_image = self.transforms(image).float()

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
    dataset = Dataset(root='',
                      phase='train',
                      input_shape=(1, 112, 112))

    
    trainloader = data.DataLoader(dataset, batch_size=4  , shuffle=False, num_workers=4)

    for index , (transformed , label , image_path) in tqdm(enumerate(trainloader)):

        answer_label = list(map(int , map(lambda x : x.split('/')[-2], image_path)))
        answer_label_tensor = torch.tensor(answer_label , dtype=torch.int32)
        print(f"Batch {index + 1} - Transformed Image Shape: {transformed.shape}, Label: {label}, answer_label: {answer_label_tensor}")

        if torch.equal(label, answer_label_tensor):
            pass

        else:
            raise ValueError("Labels do not match! label: {}, answer_label: {} img_path: {}".format(label, answer_label_tensor, image_path))
