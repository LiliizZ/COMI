import os
from PIL import Image
import PIL
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import logging
import cv2
logger = logging.getLogger(__file__)


'''class CustomDataset(Dataset):
    def __init__(self, data_root, mapping_file, transform=None):
        self.data_root = data_root
        self.mapping = self.load_mapping(mapping_file)
        self.transform = transform

        # 获取所有图像文件路径和对应的类别标签
        self.image_paths, self.labels = self.get_image_paths_and_labels()

    def load_mapping(self, mapping_file):
        mapping = {}
        with open(mapping_file, 'r') as f:
            for idx, line in enumerate(f):
                folder_name = line.strip()
                mapping[folder_name] = idx
        return mapping

    def get_image_paths_and_labels(self):
        image_paths = []
        labels = []
        for folder_name in os.listdir(self.data_root):
            folder_path = os.path.join(self.data_root, folder_name)
            
            #folder_path = os.path.join(self.data_root, folder_name, 'images')
            if not os.path.isdir(folder_path):
                continue
            label = self.mapping.get(folder_name)
            if label is not None:
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    image_paths.append(image_path)
                    labels.append(label)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label'''



# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.data_size = len(dataset)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id = self.dataset[idx][0]
        image_path = self.dataset[idx][1]
        image = Image.open(image_path).convert('RGB')
        #image = cv2.imread(image)
        if self.transform:
            image = self.transform(image)

        label = self.dataset[idx][2]

        return id, image, label


class ImageNetData(Dataset):
    def __init__(self, args, mapping_file, data_root):
        self.data_root = data_root
        self.batch_size = args.batch_size
        self.mapping = self.load_mapping(mapping_file)

        # 获取所有图像文件路径和对应的类别标签
        self.data = self.load_data()
    
    def load_mapping(self, mapping_file):
        mapping = {}
        with open(mapping_file, 'r') as f:
            for idx, line in enumerate(f):
                folder_name = line.strip()
                mapping[folder_name] = idx
        return mapping

    def load_data(self):
        results = []
        id = 0
        for folder_name in os.listdir(self.data_root):
            folder_path = os.path.join(self.data_root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label = self.mapping.get(folder_name)
            if label is not None:
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    try:
                        image = Image.open(image_path)
                        results.append([id, image_path, label])
                        id += 1
                    except (PIL.UnidentifiedImageError, FileNotFoundError) as e:
                        print(f"Error loading image {image_path}: {e}")
        return results
        
    

# 定义数据加载器
def get_data_loader(data, batch_size, transform=None, shuffle=False, num_workers=4):
    dataset = CustomDataset(data, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader



