from discriminator import Discriminator
import torch
from torch import nn
from torch import optim

class Trainer():
    def __init__(self, discriminator, generator):
        self.discriminator = discriminator
        self.generator = generator
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCELoss()
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)

    def train_discriminator(self, x):
        self.discriminator.zero_grad()
        discriminator_real_loss = self._train_discriminator_real(x)
        discriminator_fake_loss = self._train_discriminator_fake()
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

    def _train_discriminator_real(self, x):
        x_real = x.reshape(-1,784).to(self.device)
        y_real = torch.ones(x.shape[0],1).to(self.device)
        discriminator_output = self.discriminator(x_real)
        return self.criterion(discriminator_output, y_real)

    def _train_discriminator_fake(self):
        z = torch.randn(64, 32).to(self.device)
        x_fake = self.generator(z)
        y_fake = torch.zeros(64, 1).to(self.device)
        discriminator_output = self.discriminator(x_fake)
        return self.criterion(discriminator_output, y_fake)
