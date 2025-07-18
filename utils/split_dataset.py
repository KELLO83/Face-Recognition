
import os
import glob
import natsort
dataset_path = '/home/ubuntu/arcface-pytorch/dataset/ms1m-arcface'
move_dir = 'home/ubuntu/arcface-pytorch/dataset/test/ms1m-arcface'


test_ration = 0.2



folder_list = natsort.natsorted(os.listdir(dataset_path))

split_len = int(len(folder_list) * test_ration)

q = len(folder_list[split_len : ])

split_folder = folder_list[-q : ]


test_folder_path = [os.path.join(dataset_path, x) for x in split_folder]
