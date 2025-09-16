from PIL import Image
from torchvision import transforms
import torch
import os
from tqdm import tqdm

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(400),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.4711, 0.4475, 0.4080], [0.2438, 0.2390, 0.2420]),
])

img_path = 'image/train2014'
save_path = 'image/transimg2'

def inverse_normalize(tensor, mean, std):
    # 克隆tensor以避免修改原始数据
    tensor = tensor.clone().detach()
    # 逆向标准化
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # t = (t * s) + m

    return tensor

for filename in tqdm(os.listdir(img_path)):
    #print(filename)
    file = img_path +'/' + filename
    saveimg = save_path +'/'+ filename

    img = Image.open(file).convert('RGB')  # 确保图片是RGB格式

    #print(img.size)
    # 应用变换
    transformed_img = train_transform(img)

    #print(transformed_img.size())  # 输出应该是 (C, H, W) 格式，例如 (3, 32, 32)

    # inverse_norm = transforms.Normalize(
    #    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
    #    std=[1/0.2023, 1/0.1994, 1/0.2010]
    # )

    mean = torch.tensor([0.4711, 0.4475, 0.4080])
    std = torch.tensor([0.2438, 0.2390, 0.2420])
    mean = torch.tensor([0.4711, 0.4475, 0.4080])
    std = torch.tensor([0.2438, 0.2390, 0.2420])



    # 应用逆向标准化
    inverse_norm = inverse_normalize(transformed_img, mean, std)

    # 然后，将其转换为PIL图像
    transformed_img = transforms.ToPILImage()(inverse_norm)
    #print(transformed_img.size)
    transformed_img.save(saveimg)

