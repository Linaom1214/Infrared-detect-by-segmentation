import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision import transforms
from skimage.segmentation import mark_boundaries
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.transforms import Grayscale, RandomHorizontalFlip, Resize, ToTensor
import numpy as np
import matplotlib.pyplot as plt
import os


class InfraredDataset(Dataset):
    def __init__(self, dataset_dir, image_index, image_size=256):
        super(InfraredDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_index = image_index
        self.transformer = transforms.Compose([
            Resize((int(image_size), int(image_size))),
            Grayscale(),
            ToTensor(),
            RandomHorizontalFlip(0.5),
        ])

    def __getitem__(self, index):
        image_index = self.image_index[index].strip('\n')
        image_path = os.path.join(self.dataset_dir, 'images', '%s.png' % image_index)
        label_path = os.path.join(self.dataset_dir, 'masks', '%s_pixels0.png' % image_index)
        image = Image.open(image_path)
        label = Image.open(label_path)
        torch.manual_seed(1024)
        tensor_image = self.transformer(image)
        torch.manual_seed(1024)
        label = self.transformer(label)
        label[label > 0] = 1
        return tensor_image, label

    def __len__(self):
        return len(self.image_index)


if __name__ == "__main__":
    f = open('../sirst/idx_427/trainval.txt').readlines()
    ds = InfraredDataset(f)
    # 数据集测试
    for i, (image, label) in enumerate(ds):
        image, label = to_pil_image(image), to_pil_image(label)
        image, label = np.array(image), np.array(label)
        print(image.shape, label.shape)
        vis = mark_boundaries(image, label, color=(1, 1, 0))
        image, label = np.stack([image] * 3, -1), np.stack([label] * 3, -1)
        plt.imsave('image_%d.png' % i, vis)
