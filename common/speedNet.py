import torch.nn as nn
import torch
class speednet(nn.Module):
    def __init__(self):
        super(speednet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Tanh(),
            nn.Linear(2, 1)
        )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def forward(self, x):
        return self.net(x).detach().cpu().numpy()

    def learn(self, input, output):
        input = torch.FloatTensor(input).to('cuda:0')
        output = torch.FloatTensor(output).to('cuda:0')
        speed = self.net(input)
        lossf = nn.MSELoss(reduction='none')
        loss = torch.mean(lossf(speed, output))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.detach().cpu().numpy()

    def save(self, PATH):
        torch.save(self.net.state_dict(), PATH)