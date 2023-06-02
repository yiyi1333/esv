import torch
import torch.nn as nn
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            # 输入为30， 一层1000个参数的中间层， 10个输出
            nn.Linear(20, 4096, bias=True),
            # sigmoid激活函数
            nn.Sigmoid(),
            nn.Linear(4096, 10 , bias=True),
        )
        # 初始化每一层的权重参数
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.model(x)
        return x