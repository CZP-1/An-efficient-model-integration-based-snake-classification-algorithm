import torch
import torchvision
from torchvision import transforms
import json

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())
  
if __name__ == '__main__':
    trans = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.ImageFolder(root="/home/lkd22/PBVS/all_data/SAR/train", transform=trans)

    # train
    with open("/home/lkd22/PBVS/train_data/SAR_train.json", "r") as f:
        all_info = json.load(f)
    num_classes = all_info["num_classes"]
    data = all_info["annotations"]
    samples = [(item['fpath'], item['category_id']) for item in data]
    train_dataset.samples = samples
    print(len(train_dataset.samples))
    # val
    with open("/home/lkd22/PBVS/train_data/SAR_val.json", "r") as f:
        all_info = json.load(f)
    num_classes = all_info["num_classes"]
    data = all_info["annotations"]
    samples = [(item['fpath'], item['category_id']) for item in data]
    train_dataset.samples.extend(samples)
    print(len(train_dataset.samples))

    val_dataset = torchvision.datasets.ImageFolder(root="/home/data3/changhao/Datasets/PBVS2022/images_SAR", transform=trans)
    train_dataset.samples.extend(val_dataset.samples)
    print(len(train_dataset.samples))
    
    print(getStat(train_dataset))