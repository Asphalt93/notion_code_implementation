import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

class StackedAutoEncoder(nn.Module):
    def __init__(self, fine_tune=False):
        super(StackedAutoEncoder, self).__init__()
        self.fine_tune = fine_tune

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 100, 5, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(100, 150, 5, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(150, 200, 3, padding=1),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(200, 150, kernel_size=2, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(150, 100, kernel_size=2, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(100, 3, kernel_size=2, stride=2, padding=2),
            nn.Tanh()
        )

        self.fc = nn.Sequential(
            nn.Linear(7200, 300),
            nn.Linear(300, 10)
        )


    def forward(self, x):
        if self.fine_tune:
            out = self.encoder(x)
            out = out.view(-1, np.prod(out.size()[1:]))
            out = self.fc(out)
            out = F.softmax(out, dim=1)
        else:
            out = self.encoder(x)
            out = self.decoder(out)
            out = torch.sigmoid(out)
        return out




class StackedAutoEncoder_noMax(nn.Module):
    def __init__(self, fine_tune=False):
        super(StackedAutoEncoder_noMax, self).__init__()
        self.fine_tune = fine_tune

        self.encoder=nn.Sequential(
            nn.Conv2d(3, 100, 5, padding=1),
            nn.Tanh(),
            nn.Conv2d(100, 150, 5, padding=1),
            nn.Tanh(),
            nn.Conv2d(150, 200, 3, padding=1),
            nn.Tanh(),
        )

        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(200, 150, 3, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(150, 100, 3, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(100, 3, 3, 1, 1),
            nn.Tanh()
        )

        self.fc = nn.Sequential(
            nn.Linear(156800, 300),
            nn.Linear(300, 10)
        )


    def forward(self, x):
        if self.fine_tune:
            out = self.encoder(x)
            out = out.view(-1, np.prod(out.size()[1:]))
            out = self.fc(out)
            out = F.softmax(out, dim=1)
        else:
            out = self.encoder(x)
            out = self.decoder(out)
            out = torch.sigmoid(out)

        out = torch.sigmoid(out)

        return out
#
#
#
# model = StackedAutoEncoder_noMax(fine_tune=True).cuda()
# summary(model, (3,32,32))
