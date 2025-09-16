import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# 数据集路径
dataset_path = 'image'

# 数据集变换，这里只转换为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载数据集
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12)

# 初始化累加器
mean = 0.0
std = 0.0
nb_samples = 0

# 遍历数据集
for data, _ in tqdm(dataloader):
    batch_samples = data.size(0)  # 当前批次的样本数
    data = data.view(batch_samples, data.size(1), -1)  # 重新形状为[batch_size, channels, height*width]

    # 累加通道的均值和方差
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

# 计算最终的均值和方差
mean /= nb_samples
std /= nb_samples

print(f'Mean: {mean}')
print(f'Std: {std}')
