# 加载官方预训练参数，进行迁移训练
# 保存损失值最小的参数  正确率最高的参数

import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import datasets,transforms,models
import os


# 数据集地址
data_directory = "E:\\AI\\grabage"

data_transforms = {
    'train':transforms.Compose([
        transforms.Resize((230,230)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]),
    'test':transforms.Compose([
        transforms.Resize((230,230)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])  
}

trainset = datasets.ImageFolder(os.path.join(data_directory,'train'),data_transforms['train'])
testset = datasets.ImageFolder(os.path.join(data_directory,'test'),data_transforms['test'])

# 加载训练集和测试集
trainloader = torch.utils.data.DataLoader(trainset,batch_size=25,shuffle=True)
testloader = torch.utils.data.DataLoader(testset,batch_size=25,shuffle=True)

# 使用googlenet进行迁移训练
yang = models.googlenet(pretrained=True)
#print(yang)

# 冻结参数
for param in yang.parameters():
    param.requires_grade = False

yang.dropout = nn.Dropout(p=0.5,inplace=True)

# 修改输出层的神经元个数
yang.fc = nn.Linear(in_features=1024,out_features=7,bias=True)

# 使用GPU加速训练
CUDA = torch.cuda.is_available()
if CUDA :
    yang = yang.cuda()

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化方法
#optimizer = optim.SGD(yang.parameters(),lr=0.0001,momentum=0.9)
optimizer = optim.Adam(yang.parameters(),lr=0.0001)

# 加载模型及参数
def load_param(model,path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

# 保存权重参数
def save_param(model,path):
    torch.save(model.state_dict(),path,_use_new_zipfile_serialization=False)

# 训练
def train(model,criterion,optimizer,epochs=1):
    global train_num
    model.train()
    global min_loss
    for epoch in range(epochs):
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels = data
            if CUDA:
                inputs,labels = inputs.cuda(),labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()            
            optimizer.step()

            running_loss += loss.item()
                        
            if i%2==1:
                if i==1:
                    min_loss = running_loss/2
                print('[Epoch:%d,Batch:%3d] Loss: %f' % (train_num+1,i+1,running_loss/2))                
                if ((running_loss/2)<min_loss):
                    min_loss = running_loss/2
                    save_param(yang,'E:\\AI\\grabage\\model\\yang_googlenet_loss.pth')   
                running_loss = 0.0

    print('Finished Traing')

# 测试
def test(testloader,model):
    correct = 0
    total = 0
    global max_accuracy
    for data in testloader:
        images,labels = data
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
        if hasattr(torch.cuda, 'empty_cache'):
	        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    running_accuracy = torch.true_divide(100*correct,total)
    if running_accuracy>max_accuracy:
        max_accuracy = running_accuracy
        save_param(yang,'E:\\AI\\grabage\\model\\yang_googlenet_accuracy.pth')
    print('Accuracy on the test : %f %%' % (max_accuracy))
    print('*****************Finished*****************')


if __name__ == "__main__":
    max_accuracy = 0.0  
    min_loss = 0.0

    # 设置训练的次数
    for train_num in range(10):
        #load_param(yang,'E:\\AI\\grabage\\model\\yang_googlenet_loss.pth')
        train(yang,criterion,optimizer,epochs=1)    
        load_param(yang,'E:\\AI\\grabage\\model\\yang_googlenet_loss.pth')
        test(testloader,yang)