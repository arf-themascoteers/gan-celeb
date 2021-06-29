import torch
from torch import  nn
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(32, 64)
        self.lrelu = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(64, 784)

    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x.reshape(x.shape[0],-1))
        return torch.tanh(x)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)
    writer = SummaryWriter()
    for i in range(5):
        data = torch.randn((64,32))
        y_pred = model(data)
        image = y_pred.reshape(64,1,28,28)
        writer.add_image(f"Generated{i}", image[0])

