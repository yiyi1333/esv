import torch.cuda
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import GoogLeNet_Weights

from dataset import dataset_build
from net import VGG_16

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
model = torchvision.models.densenet201(pretrained=True)
model.classifier = torch.nn.Linear(1920, class_nums, bias=True)

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
epoch = 101

# tensorboard, 指定日志名称
writer = SummaryWriter("../logs/densnet201/5class2/")

for i in range(epoch):
    print("--------------------第{}轮训练开始-------------------".format(i + 1))

    # 训练过程
    model.train()
    for data in train_loader:
        imgs, targets = data
        if gpu_available:
            imgs = imgs.cuda()
            targets = targets.cuda()
        # print(targets)
        outputs = model(imgs)
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
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 创建一个10 * 10的矩阵，用于记录预测结果
    confusion_matrix = torch.zeros(class_nums, class_nums, dtype=torch.int64)

    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if gpu_available:
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model(imgs)
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
        frrs.append(frr)

    # 计算平均Precision, Recall, Frr
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_frr = sum(frrs) / len(frrs)

    # 计算F1-score
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    print("第{}轮测试，平均Precision为{}, 平均Recall为{}, 平均F1-score为{},平均FRR为{}".format(i + 1, avg_precision,
                                                                                              avg_recall, avg_f1,
                                                                                              avg_frr))
    writer.add_scalar("Precision", avg_precision, total_test_step)
    writer.add_scalar("Recall", avg_recall, total_test_step)
    writer.add_scalar("F1", avg_f1, total_test_step)
    writer.add_scalar("FRR", avg_frr, total_test_step)

    # 记录测试损失
    print("第{}轮测试，损失为{}".format(i + 1, total_test_loss))
    print("第{}轮测试，准确率为{}".format(i + 1, total_accuracy / len(test_dataset)))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_acc", total_accuracy / len(test_dataset), total_test_step)


    # 保存模型
    if i % 10 == 0:
        torch.save(model.state_dict(), "../model/densnet201/densnet201_{}.pth".format(i))
    print("\n")

writer.close()

