import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
from PIL import Image
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# 设置超参数
#每次的个数
BATCH_SIZE = 20
#迭代次数
EPOCHS = 10
#采用cpu还是gpu进行计算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
# transforms.Resize(100): 将图像的短边调整为100像素，长边按比例缩放。
# transforms.RandomVerticalFlip(): 以50%的概率对图像进行垂直翻转。
# transforms.RandomCrop(50): 在图像上随机选择一个50x50的区域并裁剪。
# transforms.RandomResizedCrop(150): 首先，对图像进行缩放和长宽比变换，然后裁剪一个150x150的区域。这是一种常见的数据增强策略。
# transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5): 随机改变图像的亮度、对比度和色调。
# transforms.ToTensor(): 将PIL图像或NumPy ndarray转换为torch.Tensor。它会将图像的像素强度值从0-255缩放到0-1之间。
# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]): 对图像进行归一化。这里的两个参数分别是三个通道的平均值和标准差。这会使图像的像素强度值变为-1到1之间。
transform = transforms.Compose([
    transforms.Resize(100),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(50),
    transforms.RandomResizedCrop(150),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
#导入训练数据
dataset_train = datasets.ImageFolder('D:\\pythonProject\\train', transform)

#导入测试数据
dataset_test = datasets.ImageFolder('D:\\pythonProject\\test', transform)

test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

# print(dataset_train.imgs)
# print(dataset_train[0])
# print(dataset_train.classes)
classess=dataset_train.classes #标签
class_to_idxes=dataset_train.class_to_idx #对应关系
print(class_to_idxes)
# print(dataset_train.class_to_idx)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
# for batch_idx, (data, target) in enumerate(train_loader):
#     # print(data)
#     print(target)
#     data, target = data.to(device), target.to(device).float().unsqueeze(1)
#     # print(data)
#     print(target)

# 定义网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #卷积层conv，池化层pool
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

modellr = 1e-4

# 实例化模型并且移动到GPU

model = ConvNet().to(device)
print(model)
# 选择简单暴力的Adam优化器，学习率调低

optimizer = optim.Adam(model.parameters(), lr=modellr)
#调整学习率
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 5))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        output = model(data)

        # print(output)

        loss = F.binary_cross_entropy(output, target)

        loss.backward()

        optimizer.step()

        if (batch_idx + 1) % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                       100. * (batch_idx + 1) / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            # print(target)
            output = model(data)
            # print(output)
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
            pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)
            correct += pred.eq(target.long()).sum().item()

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))




# 训练
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

torch.save(model, 'model_insects0.pth')
