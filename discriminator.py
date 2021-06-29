import torch
from torch import  nn
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn1 = nn.Conv2d(1,8,kernel_size=(5,5))
        self.lrelu = nn.LeakyReLU(0.2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn2 = nn.Conv2d(8,16,kernel_size=(4,4))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.lrelu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.lrelu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    mnist_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset=mnist_data,
                                              batch_size=64,
                                              shuffle=True)

    model = Discriminator().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    num_epochs = 5
    for epoch  in range(0, 5):
        for data, y_true in dataloader:
            optimizer.zero_grad()
            y_pred = model(data)
            loss = F.nll_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
