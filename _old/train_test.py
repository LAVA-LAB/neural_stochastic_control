import torch
from torch import nn
import torch.nn.functional as F

class CertificateNetwork(nn.Module):

    def __init__(self, in_features=2, h1=5, h2=5, out_features=1):
        super().__init__()  # Instantiate nn.module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.out(x))

        return x

net = CertificateNetwork(in_features = 2)

params = list(net.parameters())

input = torch.randn(2)
output = net(input)
target = torch.randn(1)  # a dummy target, for example
criterion = nn.MSELoss()

print(list(net.fc1.parameters()))

##

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.1)

# in your training loop:
for i in range(1000):
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    # print(output)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update

print(list(net.fc1.parameters()))


assert False




criterion = nn.MSELoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(certificate_net.parameters(), lr=learning_rate)
max_iter = 2000

for iteration in range(max_iter):

    out = certificate_net(torch.tensor([0.5, -10]))
    print(out)

    loss = criterion(out, torch.ones_like(out))

    certificate_net.zero_grad()
    loss.backward()
    print(certificate_net.fc1.bias.grad)
    optimizer.step_train(,
