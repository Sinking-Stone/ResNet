import torch
import numpy
import time
import os,sys
import matplotlib.pyplot as plt
from resnet import ResNet20Test
# from transformer_resnet import ResNet20
from MyCode.ResNet20.cifar10.read_data import read_cifar10
import os

# 将模型保存为txt文件
def savetxt():
    pm = torch.load('../cnn/91.74best_acc.pth')['model']

    fp = os.open("../parameters1.txt", os.O_RDWR|os.O_CREAT)
    numpy.set_printoptions(threshold=sys.maxsize)   # 不让数据中出现...
    for key, item in pm.items():
        os.write(fp, key.encode())
        pn  = numpy.array2string(item.numpy(), separator=',').encode()
        os.write(fp, pn)
        os.write(fp, "\n".encode())
    os.close(fp)

# 验证正确性，执行之后会保存一个example.txt文件，只保存一张图的数据
def testdata():
    
    data_dir = '../data'
    batchsize = 2

    _,data_loader_test = read_cifar10(batchsize,data_dir)

    # 保存数据参数
    # pm=torch.load('./cnn/91.74best_acc.pth')["model"]
    # fp = os.open("./data/parameters.txt", os.O_RDWR|os.O_CREAT)
    # numpy.set_printoptions(threshold=sys.maxsize)   # 不让数据中出现...
    # for key, item in pm.items():
    #     os.write(fp, key.encode())
    #     pn  = numpy.array2string(item.cpu().numpy(), separator=',').encode()
    #     os.write(fp, pn)
    #     os.write(fp, "\n".encode())
    # os.close(fp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ResNet20().to(device)
    model = ResNet20Test().to(device)
    print(model)

    state_dict = torch.load('../cnn/91.74best_acc.pth')
    model.load_state_dict(state_dict['model'], False)
    model.eval()

    infimg, inflab=[],[]
    
    with torch.no_grad():
        for data in data_loader_test:
            x_test, label_test = data
            x_test, label_test = x_test.to(device), label_test.to(device)
            infimg.append(x_test)
            inflab.append(label_test)

            # 打印出对应的图像信息
            # print("x test size", x_test.size())
            # image = x_test[0].cpu().numpy()
            # if image.shape[0] == 3:
            #     image = numpy.transpose(image, (1, 2, 0))
            # elif image.shape[0] == 1:
            #     image = image.squeeze(0)
            # plt.imshow(image)
            # plt.show()

        # print(len(infimg))
        for i in range(1):
            output = model(infimg[i])
            x,pred = torch.max(output.data,1)
            res = (pred == inflab[i]).sum().item()
            print("x: ",x)
            print("pred: ", pred, "inflab[i]: ", inflab[i], "res: ", res)


# 推理，查看其准确性
def testCode():
    since = time.time()
    data_dir = './data'
    batchsize = 128

    data_loader_train,data_loader_test = read_cifar10(batchsize,data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ResNet20().to(device)
    model = ResNet20Test().to(device)
    print(model)

    state_dict = torch.load('../cnn/91.74best_acc.pth')
    model.load_state_dict(state_dict['model'], False)
    model.eval()
    
    testing_correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader_test:
            x_test, label_test = data
            x_test, label_test = x_test.to(device), label_test.to(device)
            outputs = model(x_test)
            _,pred = torch.max(outputs.data,1)
            # print(pred)
            total += label_test.size(0)
            testing_correct += (pred == label_test).sum().item()

        print("testing_correct: "+str(testing_correct)+"\ttotal: "+str(total))
        print('Test acc: {:.4}%'.format(100*testing_correct/total))

        for data in data_loader_train:
            x_test, label_test = data
            x_test, label_test = x_test.to(device), label_test.to(device)
            outputs = model(x_test)
            _,pred = torch.max(outputs.data,1)
            # print(pred)
            total += label_test.size(0)
            testing_correct += (pred == label_test).sum().item()

    print("testing_correct: "+str(testing_correct)+"\ttotal: "+str(total))
    print('Test acc: {:.4}%'.format(100*testing_correct/total))
    # print("Loss :{:.4f}, Train acc :{.4f}, Test acc :{.4f}".format(training_loss/len(data_train),100*training_correct/len(data_train),100*testing_correct/len(data_test)))

    time_used = time.time() - since
    print("推理时间：",time_used)

if __name__ == '__main__':
    # savetxt()
    testdata()
