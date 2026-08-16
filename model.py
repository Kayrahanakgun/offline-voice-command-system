import torch
import torch.nn as nn
import torch.nn.functional as F

import preprocess

from dataset import labels, inputs

final_features = preprocess.preprocess_audio1("data/Test/TestAudio.wav")
x = torch.tensor(final_features, dtype=torch.float32)

x = x.unsqueeze(0)   # channel dimension
x = x.unsqueeze(0)   # batch dimension
print("Raw------------------------------")
print(x.shape)
print(len(x))
print("Raw------------------------------")

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

        self.fc1 = nn.LazyLinear(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        #print("Conv1:", c1.shape)
        #print("------------------------------")

        s2 = F.max_pool2d(c1, 2)
        #print("Pool1:", s2.shape)
        #print("------------------------------")

        c3 = F.relu(self.conv2(s2))
        #print("Conv2:", c3.shape)
        #print("------------------------------")

        s4 = F.max_pool2d(c3, 2)
        #print("Pool2:", s4.shape)
        #print("------------------------------")

        s4 = torch.flatten(s4, 1)
        #print("Flatten:", s4.shape)
        #print("------------------------------")

        f5 = F.relu(self.fc1(s4))
        f5 = self.dropout(f5)
        output = self.fc2(f5)

        #print("Output------------------------")
        #print(output.shape)
        #print("------------------------------")

        return output


net = Net()
print(net)
output = net(x)

#print(output)


