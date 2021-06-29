import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


transform = transforms.ToTensor()
mnist_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

writer = SummaryWriter()
for data, target in data_loader:
    for i in range(5):
        image = data[i]
        writer.add_image(f"Image-{i}", image)