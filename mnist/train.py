import torch
import time
from torch import nn

from tensorboardX import SummaryWriter

from resnet import ResNet20Test
from read_data import read_mnist
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1,2'

def main():
    since = time.time()
    data_dir = '../data'
    model_dir = '../cnn'
    batchsize = 32
    n_epochs = 200
    best_acc = 0.0#记录测试最高准确率
    Lr = 0.1

    data_loader_train,data_loader_test = read_mnist(batchsize,data_dir)

    device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model = ResNet20Test().to(device)
    print(model)
    #取出权重参数和偏置参数，仅对权重参数加惩罚系数
    weight_p, bias_p = [],[]
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p +=[p]
        else:
            weight_p +=[p]
    cost = nn.CrossEntropyLoss().to(device)      #Q3：要不要加正则项

    #L1正则
    # regularization_loss = 0
    # for param in model.parameters():
    #     regularization_loss += torch.sum(torch.abs(param))
    #
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD([{'params':weight_p,'weight_decay':1e-4},
                            {'params':bias_p,'weight_decay':0}],lr=Lr,momentum=0.9)#内置的SGD是L2正则，且对所有参数添加惩罚，对偏置加正则易导致欠拟合，一般只对权重正则化
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[82,122,163],gamma=0.1,last_epoch=-1)#Q4: 对应多少步,epoch= 32000/(50000/batch_size),48000,64000

    Loss_list = []
    Accuracy_list = []
    
    writer = SummaryWriter()
    for epoch in range(n_epochs):

        model.train()

        training_loss = 0.0
        training_correct = 0
        training_acc = 0.0
        print("Epoch {}/{}".format(epoch+1,n_epochs))
        print("-"*30)

        total_train = 0
        for i,data in enumerate(data_loader_train):
            x,labels = data
            x,labels = x.to(device), labels.to(device)

            # print(x.shape)
            # print(label.shape)
            #前向传播计算损失
            outputs = model(x)
            loss = cost(outputs, labels)
            training_loss += loss.item()
            # print(outputs)
            _,pred = torch.max(outputs,1)#预测最大值所在位置标签
            total_train += labels.size(0)
            num_correct = (pred == labels).sum()
            training_acc += num_correct.item()

            writer.add_scalar('Train/Loss', loss.item(), epoch*len(data_loader_train)+i)
            writer.add_scalar('Train/Acc', 100*num_correct.item()/labels.size(0), epoch*len(data_loader_train)+i)
            writer.add_scalar('Train/Total_Acc', 100*training_acc/total_train, epoch*len(data_loader_train)+i)
            
            #反向传播+优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('Epoch：', epoch, 'train loss:', training_loss/len(data_loader_train))
        # Loss_list.append(training_loss/len(data_loader_train))
            if i % 100 == 99:
                print('[%d, %5d] traing_loss: %f' % (epoch + 1, i + 1, training_loss / 100))
                Loss_list.append(training_loss / 100)
                training_loss = 0.0
        print('Train acc:{:.4}%'.format(100*training_acc/total_train))

        scheduler.step()

        model.eval()
        testing_correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader_test:
                x_test, label_test = data
                x_test, label_test = x_test.to(device), label_test.to(device)
                outputs = model(x_test)
                _,pred = torch.max(outputs.data,1)
                total += label_test.size(0)
                testing_correct += (pred == label_test).sum().item()
        print('Test acc: {:.4}%'.format(100*testing_correct/total))
        Accuracy_list.append(100*testing_correct/total)
        acc = 100*testing_correct/total
        if acc>best_acc:
            best_acc = acc
            best_acc_loc = epoch
        # print("Loss :{:.4f}, Train acc :{.4f}, Test acc :{.4f}".format(training_loss/len(data_train),100*training_correct/len(data_train),100*testing_correct/len(data_test)))

    print('test best acc:{}% at epoch{}'.format(best_acc,best_acc_loc))
    time_used = time.time() - since
    print('-' * 30)
    print('训练用时： {:.0f}m {:.0f}s'.format(time_used // 60, time_used % 60))
    # print('最高准确率: {}%'.format(100 * best_acc))

    #保存参数优化器等
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, os.path.join(model_dir,'{}best_acc_mnist.pth'.format(best_acc)))

if __name__ == '__main__':
    main()
