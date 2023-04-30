import os

import torch.cuda
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import dataset_build
from net import VGG_16, DNN

# 超参
batch_size = 64 # 一次训练的样本数
learning_rate = 0.001 # 学习率
class_nums = 10 # 分类数

if torch.cuda.is_available():
    #打印GPU信息
    print(torch.cuda.get_device_name(0))
gpu_available = torch.cuda.is_available()

# 准备数据集
train_dataset = dataset_build.load("../dataset", True)
test_dataset = dataset_build.load("../dataset", False)

# 准备数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 准备模型VGG
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.classifier[6] = torch.nn.Linear(4096, class_nums, bias=True)
# 加载预训练参数
vgg16.load_state_dict(torch.load("../model/result/vgg16_5_2.pth"))
# 准备模型GoogleNet
google_net = torchvision.models.googlenet(pretrained=True)
google_net.fc = torch.nn.Linear(1024, class_nums, bias=True)
google_net.load_state_dict(torch.load("../model/result/googlenet_v1_5_2.pth"))
# 准备模型ResNet
res_net = torchvision.models.resnet18(pretrained=True)
res_net.fc = torch.nn.Linear(512, class_nums, bias=True)
res_net.load_state_dict(torch.load("../model/result/resnet18_5_2.pth"))
# 准备主干网络
dnn = DNN.DNN()

# 模型放入GPU
if gpu_available:
    vgg16 = vgg16.cuda()
    google_net = google_net.cuda()
    res_net = res_net.cuda()
    dnn = dnn.cuda()

# 损失函数： 交叉熵损失函数
loss_func = torch.nn.CrossEntropyLoss()
if gpu_available:
    loss_func = loss_func.cuda()

# 优化器
optimzer = torch.optim.SGD(dnn.parameters(), lr=learning_rate)

# 训练
total_train_step = 0
total_test_step = 0
epoch = 101

# tensorboard, 指定日志名称
writer = SummaryWriter("../logs/dnn/")

for i in range(epoch):
    print("--------------------第{}轮训练开始-------------------".format(i + 1))

    # 训练过程
    dnn.train()
    for data in train_loader:
        imgs, targets = data
        if gpu_available:
            imgs = imgs.cuda()
            targets = targets.cuda()

        vgg16_features = vgg16(imgs)
        google_net_features = google_net(imgs)
        res_net_features = res_net(imgs)
        # 将上述三个网络的特征拼接起来
        features = torch.cat((vgg16_features, google_net_features, res_net_features), 1)
        # print('vgg16 shape: ' + str(vgg16_features.size()))
        # print('concat shape: ' + str(features.size()))
        outputs = dnn(features)
        loss = loss_func(outputs, targets)

        # 反向传播
        optimzer.zero_grad()
        loss.backward()

        # 更新参数
        optimzer.step()

        # 记录训练损失
        total_train_step += 1
        if total_train_step % 10 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试过程
    dnn.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 创建一个n * n的矩阵，用于记录预测结果
    confusion_matrix = torch.zeros(class_nums, class_nums, dtype=torch.int64)

    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if gpu_available:
                imgs = imgs.cuda()
                targets = targets.cuda()

            vgg16_features = vgg16(imgs)
            google_net_features = google_net(imgs)
            res_net_features = res_net(imgs)
            # 将上述三个网络的特征拼接起来
            features = torch.cat((vgg16_features, google_net_features, res_net_features), 1)
            # print(targets)
            outputs = dnn(features)
            # loss
            loss = loss_func(outputs, targets)
            total_test_loss += loss.item()
            # acc
            accuracy = (outputs.argmax(1) == targets).sum().item()
            # 预测结果
            results = outputs.argmax(1)
            # 遍历预测结果
            for t, p in zip(targets.view(-1), results.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            total_accuracy += accuracy

    total_test_step += 1

    # 打印混淆矩阵
    # print(confusion_matrix)

    # 计算每一类的Precision, Recall
    precisions = []
    recalls = []
    frrs = []
    for k in range(class_nums):
        # 计算每一类的Precision tp / (tp + fp)
        precision = confusion_matrix[k][k] / confusion_matrix[:, k].sum()
        # 计算每一类的Recall tp / (tp + fn)
        recall = confusion_matrix[k][k] / confusion_matrix[k, :].sum()
        # 计算每一类的FRR fn / (fn + tp)
        frr = (confusion_matrix[k, :].sum() - confusion_matrix[k][k]) / confusion_matrix[k, :].sum()
        precisions.append(precision)
        recalls.append(recall)


    # 计算平均Precision, Recall, Frr
    if len(precisions) == 0:
        avg_precision = 0
    else:
        avg_precision = sum(precisions) / len(precisions)
    if len(recalls) == 0:
        avg_recall = 0
    else:
        avg_recall = sum(recalls) / len(recalls)
    if len(frrs) == 0:
        avg_frr = 0
    else:
        avg_frr = sum(frrs) / len(frrs)

    # 计算F1
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    print("第{}轮测试，平均Precision为{}, 平均Recall为{}, 平均F1-score为{},平均FRR为{}".format(i + 1, avg_precision, avg_recall, avg_f1, avg_frr))
    writer.add_scalar("Precision", avg_precision, total_test_step)
    writer.add_scalar("Recall", avg_recall, total_test_step)
    writer.add_scalar("F1", avg_f1, total_test_step)
    writer.add_scalar("FRR", avg_frr, total_test_step)

    # 记录测试损失
    print("第{}轮测试，损失为{}".format(i + 1, total_test_loss))
    print("第{}轮测试，准确率为{}".format(i + 1, total_accuracy / len(test_dataset)))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_acc", total_accuracy / len(test_dataset), total_test_step)

    # 如果目录不存在，则创建目录
    if not os.path.exists('../model/dnn/'):
        os.makedirs('../model/dnn/')
    # 保存模型
    if i % 10 == 0:
        torch.save(dnn.state_dict(), "../model/dnn/dnn__{}.pth".format(i))
    print("\n")

writer.close()
