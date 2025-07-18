import kagglehub


# Download latest version
path = kagglehub.dataset_download("yakhyokhuja/ms1m-arcface-dataset")

print("Path to dataset files:", path)