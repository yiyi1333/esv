import numpy as np
import torch.cuda
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from dataset.dataset_build import SigComp2011_Dataset_Chinese

from net import VGG_16
from dataset import dataset_build
from torch.utils.data import DataLoader
# 超参
batch_size = 16 # 一次训练的样本数
learning_rate = 0.001 # 学习率

if torch.cuda.is_available():
    #打印GPU信息
    print(torch.cuda.get_device_name(0))
gpu_available = torch.cuda.is_available()

# 准备数据集
train_dataset = dataset_build.load("../dataset", True, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
test_dataset = dataset_build.load("../dataset", False)

# 准备数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 准备模型VGG
model = torchvision.models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 40, bias=True)
# model = VGG_16.VGG16_3()
# 模型放入GPU
if gpu_available:
    model = model.cuda()

# 损失函数： 交叉熵损失函数
loss_func = torch.nn.CrossEntropyLoss()
if gpu_available:
    loss_func = loss_func.cuda()

# 优化器
optimzer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练
total_train_step = 0
total_test_step = 0
epoch = 100

# tensorboard
writer = SummaryWriter("../logs/vgg16_3/")

for i in range(epoch):
    print("--------------------第{}轮训练开始-------------------".format(i + 1))

    # 训练过程
    for data in train_loader:
        imgs, targets = data
        if gpu_available:
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = model(imgs)
        loss = loss_func(outputs, targets)

        # 反向传播
        optimzer.zero_grad()
        loss.backward()

        # 更新参数
        optimzer.step()

        # 记录训练损失
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("第{}步训练，损失为{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试过程
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if gpu_available:
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_func(outputs, targets)
            total_test_loss += loss.item()

    # 记录测试损失
    print("第{}轮测试，损失为{}".format(i + 1, total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(model.state_dict(), "../model/vgg16_3/vgg16_3_{}.pth".format(i + 1))
    print("\n")


writer.close()


