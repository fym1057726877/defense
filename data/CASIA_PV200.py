import os
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset
from utils import get_project_path

IMAGE_SIZE = 64

transform_train = transforms.Compose([
    transforms.Resize(size=IMAGE_SIZE),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.432], std=[0.032])
])

transform_test = transforms.Compose([
    transforms.Resize(size=IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.432], std=[0.032])
])


class CASIA_PV200(Dataset):
    def __init__(self, transform):
        super().__init__()
        path_dir_train = os.path.join(get_project_path(project_name="Defense"), "data", "CASIA_PV200", "train")
        path_dir_val = os.path.join(get_project_path(project_name="Defense"), "data", "CASIA_PV200", "val")
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), ])
        train_set = datasets.ImageFolder(path_dir_train, transform=transform)
        val_set = datasets.ImageFolder(path_dir_val, transform=transform)

        dataset = {}
        merged_label = []
        merged_data = []
        for i in range(len(train_set)):
            data, label = train_set[i]
            if label not in dataset:
                dataset[label] = [data]
            else:
                dataset[label].append(data)

        for i in range(len(val_set)):
            data, label = val_set[i]
            if label not in dataset:
                dataset[label] = [data]
            else:
                dataset[label].append(data)

        for key in dataset.keys():
            imgs = dataset[key]
            for j in range(len(imgs)):
                merged_data.append(imgs[j])
                merged_label.append(key)

        self.data = merged_data
        self.label = merged_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label


def get_CASIA_PV200_Dataloader(train=True, batch_size=16, shuffle=False, transform=None):
    if transform is None:
        if train:
            transform = transform_train
        else:
            transform = transform_test
    dataset = CASIA_PV200(transform=transform)
    num_classes = 200
    num_images_per_class = 6
    train_images_per_class = 4
    indices = []

    # 遍历每个类别
    for class_idx in range(num_classes):
        start_idx = class_idx * num_images_per_class
        split_index = start_idx + train_images_per_class
        end_idx = (class_idx + 1) * num_images_per_class
        if train:
            subset = Subset(dataset, range(start_idx, split_index))
        else:
            subset = Subset(dataset, range(split_index, end_idx))
        indices.extend(subset.indices)
    dataset = Subset(dataset, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


trainloader = get_CASIA_PV200_Dataloader(train=True, batch_size=20, shuffle=True, transform=transform_train)
testloader = get_CASIA_PV200_Dataloader(train=False, batch_size=20, shuffle=False, transform=transform_test)

# x, y = next(iter(testloader))
# print(x.shape, y)

# def compute_mean_variance():
#     imgs = CASIA_PV200().data
#     mean, std = 0., 0.
#     for i in range(len(imgs)):
#         mean += imgs[i].mean()
#         std += imgs[i].std()
#     mean /= len(imgs)
#     std /= len(imgs)
#     print(mean, std)

# compute_mean_variance()