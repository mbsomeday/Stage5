from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch, os
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image


ds_dict = {
    'D1': r'D:\my_phd\dataset\Stage3\D1_ECPDaytime',
    'D2': r'D:\my_phd\dataset\Stage3\D2_CityPersons',
    'D3': r'D:\my_phd\dataset\Stage3\D3_ECPNight',
    'D4': r'D:\my_phd\dataset\Stage3\D4_BDD100K',
}



class pedCls_Dataset(Dataset):
    '''
        读取多个数据集的数据
    '''

    def __init__(self, ds_name_list, txt_name, img_size, get_num=None):
        self.base_dir_list = [ds_dict[ds_name] for ds_name in ds_name_list]
        self.txt_name = txt_name
        self.image_transformer = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Scales data into [0,1]
            transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        ])
        self.get_num = get_num
        self.images, self.labels = self.initImgLabel()

    def initImgLabel(self):
        '''
            读取图片 和 label
        '''
        images = []
        labels = []

        for base_dir in self.base_dir_list:
            txt_path = os.path.join(base_dir, 'dataset_txt', self.txt_name)
            with open(txt_path, 'r') as f:
                data = f.readlines()

            for line in data:
                if self.get_num is not None:
                    if len(images) >= self.get_num:
                        break
                line = line.replace('\\', os.sep)
                line = line.strip().split()
                image_path = os.path.join(base_dir, line[0])
                label = line[-1]
                images.append(image_path)
                labels.append(label)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label = self.labels[idx]
        label = np.array(label).astype(np.int64)
        img = Image.open(image_name)  # PIL image shape:（C, W, H）
        img = self.image_transformer(img)
        return img, label

def load_transformed_dataset(img_size=64, batch_size=128) -> DataLoader:
    # Load dataset and perform data transformations
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.ImageFolder(root="./stanford_cars/car_data/car_data/train", transform=data_transform)

    test = torchvision.datasets.ImageFolder(root="./stanford_cars/car_data/car_data/test", transform=data_transform)

    dataset = torch.utils.data.ConcatDataset([train, test])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def show_tensor_image(image):
    # Reverse the data transformations
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))
    plt.show()
