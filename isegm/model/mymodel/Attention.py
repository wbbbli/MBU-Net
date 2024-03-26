import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, inpchannel, oupchannel):
        super(SelfAttention, self).__init__()
        self.inpchannel = inpchannel
        self.oupchannel = oupchannel
        self.keymar = nn.Sequential(
            nn.Conv2d(in_channels=inpchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=oupchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU()
        )
        self.querymar = nn.Sequential(
            nn.Conv2d(in_channels=inpchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=oupchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU()
        )
        self.valuemar = nn.Sequential(
            nn.Conv2d(in_channels=inpchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=oupchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU()
        )
    def forward(self, x):
        batch_size = x.size(0)
        query = self.querymar(x).view(batch_size, self.oupchannel, -1)
        query = query.permute(0, 2, 1)
        key = self.keymar(x).view(batch_size, self.oupchannel, -1)
        value = self.valuemar(x).view(batch_size, self.oupchannel, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.oupchannel ** -.5) * sim_map
        sim_map = nn.functional.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.oupchannel, *x.size()[2:])

        return context

class NewAttention(nn.Module):
    def __init__(self, inpchannel, oupchannel, orichannel):
        super(NewAttention, self).__init__()
        self.inpchannel = inpchannel
        self.oupchannel = oupchannel
        self.r_mar = nn.Sequential(
            nn.Conv2d(in_channels=inpchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=oupchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU()
        )
        self.b_mar = nn.Sequential(
            nn.Conv2d(in_channels=inpchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=oupchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU()
        )
        self.c_mar = nn.Sequential(
            nn.Conv2d(in_channels=inpchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=oupchannel, out_channels=oupchannel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oupchannel),
            nn.ReLU()
        )
        self.other_mar = nn.Sequential(
            nn.Conv2d(in_channels=orichannel, out_channels=32,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
    def forward(self, feat, otherfeat):
        feat = feat.permute(0, 1, 3, 2)
        r = self.r_mar(feat)
        b = self.b_mar(feat)
        c = self.c_mar(feat)
        otherfeat = self.other_mar(otherfeat)

        ave_r = torch.mean(r, dim=1).unsqueeze(dim=1)
        subtra_r = torch.matmul(otherfeat, r-ave_r)
        covsr = torch.mean(torch.matmul(subtra_r.permute(0, 1, 3, 2), subtra_r), dim=1).unsqueeze(dim=1)
        softmax1 = nn.Softmax(dim=2)
        hsr = softmax1(covsr)

        ave_c = torch.mean(c, dim=1).unsqueeze(dim=1)
        subtra_c = torch.matmul(c-ave_c, otherfeat)
        covsc = torch.mean(torch.matmul(subtra_c, subtra_c.permute(0, 1, 3, 2)), dim=1).unsqueeze(dim=1)
        softmax2 = nn.Softmax(dim=3)
        hsc = softmax2(covsc)
        #  W*W *  W*H * H*H
        feat = torch.matmul(torch.matmul(hsc, b), hsr)

        return feat
