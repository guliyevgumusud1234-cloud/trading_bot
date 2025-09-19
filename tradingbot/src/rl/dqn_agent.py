import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

model = DQN(input_dim=6, output_dim=4)  # 6 input (fiyat, gösterge vs.) 4 output (action: long, short, kapat, bekle)
optimizer = optim.Adam(model.parameters(), lr=0.001)