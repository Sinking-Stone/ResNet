import torch
import torch.nn as nn
import numpy,sys
import torch.nn.functional as F
from torch.autograd import Function

class MyReLUx8(Function):
    @staticmethod
    def forward(ctx,input):
        output=0.16762146597970437 + 0.5026949100647794*input + 0.29657670788377677*(input**2) - 0.001188613908041178 * (input**3) - 0.0257634922669346 * (input**4) + 0.00012391328344872537 * (input**5) + 0.001239304784344458 * (input**6) -3.5488943200047796*(10**(-6)) * (input**7) + 2.1546778339421476 *(10**(-5)) * (input**8) 
        ctx.save_for_backward(output)
        return output
    @staticmethod
    def backward(ctx, grad_outputs):  # dloss/dx = dloss/dy * !(x > 0) * (dloss/dy > 0)
        output = ctx.saved_tensors[0]  # dloss/dy 
        return grad_outputs * (0.5026949100647794 + 2* 0.29657670788377677*output - 3*0.001188613908041178 * (output**2) - 4*0.0257634922669346 * (output**3) + 5* 0.00012391328344872537 * (output**4) + 6* 0.001239304784344458 * (input**5) - 7*3.5488943200047796 *(10**(-5)) * (input**6))+ 8*2.1546778339421476 *(10**(-5)) * (input**7) 

class MyReLUx7(Function):
    @staticmethod
    def forward(ctx,input):
        output=0.21285265609372403 + 0.49615019559421336*input + 0.23092606415211328*(input**2) + 0.0016979991829962182 * (input**3) - 0.0113554855294646113 * (input**4) - 0.00017701732489355438 * (input**5) + 0.00023720721467412688 * (input**6) + 5.069810781113427*(10**(-6)) * (input**7)
        ctx.save_for_backward(output)
        return output
    @staticmethod
    def backward(ctx, grad_outputs):  # dloss/dx = dloss/dy * !(x > 0) * (dloss/dy > 0)
        output = ctx.saved_tensors[0]  # dloss/dy 
        return grad_outputs * (0.49615019559421336 + 2*0.23127375301201386*output + 3*0.0016979991829962182 * (output**2) - 4*0.0113554855294646113 * (output**3) - 5* 0.00017701732489355438 * (output**4) + 6*0.00023720721467412688 * (input**5) + 7*5.069810781113427*(10**(-6)) * (input**6))

class MyReLUx6(Function):
    @staticmethod
    def forward(ctx,input):
        output=0.21317398638233181 + 0.5025652818689385*input + 0.23127375301201386*(input**2) - 0.0006165307472494883 * (input**3) - 0.011304395652213762 * (input**4) + 2.71653728139027*(10**(-5)) * (input**5) + 0.00023543278507957879 * (input**6)
        ctx.save_for_backward(output)
        return output
    @staticmethod
    def backward(ctx, grad_outputs):  # dloss/dx = dloss/dy * !(x > 0) * (dloss/dy > 0)
        output = ctx.saved_tensors[0]  # dloss/dy 
        return grad_outputs * (0.5025652818689385 + 2*0.23127375301201386*output - 3*0.0006165307472494883 * (output**2) - 4*0.011304395652213762 * (output**3) - 5* 2.71653728139027*(10**(-5)) * (output**4) + 6*0.00023543278507957879 * (input**5))

class MyReLULayer(nn.Module):
    def __init__(self):
        super(MyReLULayer, self).__init__()
    def forward(self, input):
        return MyReLUx8.apply(input)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            # MyReLULayer(),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        # print("pre:", out)
        # out = MyReLU.apply(out)
        out = F.relu(out)
        # print("aft:", out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 3, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNetTestSavefile(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetTestSavefile, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.myReLU1 = nn.Sequential(
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.myReLU2 = nn.Sequential(
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut0 = nn.Sequential()
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.myReLU3 = nn.Sequential(
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut1 = nn.Sequential()
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.myReLU4 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut2 = nn.Sequential()
        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU5 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut3 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(32)
                )
        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU6 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut4 = nn.Sequential()
        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU7 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut5 = nn.Sequential()
        self.conv13 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU8 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut6 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64)
                )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU9 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut7 = nn.Sequential()
        self.conv17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU10 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut8 = nn.Sequential()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        numpy.set_printoptions(threshold=sys.maxsize)
        with open('./data/example_cifar10.txt', 'a') as f:
            out = self.conv0(x)

            out1 = self.conv1(out)
            # out1 = MyReLU.apply(out1)
            # temp_str = numpy.array2string(out1.cpu().numpy(), separator=',').encode()

            print("第一层relu之前的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.myReLU1(out1)
            print("第一层relu之后的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.conv2(out1)
            out = out1 + self.shortcut0(out)
            print("第二层relu之前的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)
            out = F.relu(out)
            print("第二层relu之后的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)

            out1 = self.conv3(out)
            print("第三层relu之前的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.myReLU2(out1)
            print("第三层relu之前的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.conv4(out1)
            out = out1 + self.shortcut1(out)
            print("第四层relu之前的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)
            out = F.relu(out)
            print("第四层relu之后的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)

            out1 = self.conv5(out)
            print("第五层relu之前的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.myReLU3(out1)
            print("第五层relu之后的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.conv6(out1)
            out = out1 + self.shortcut2(out)
            print("第六层relu之前的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)
            out = F.relu(out)
            print("第六层relu之后的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)

            out1 = self.conv7(out)
            print("第⑦层relu之前的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.myReLU4(out1)
            print("第⑦层relu之后的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.conv8(out1)
            out = out1 + self.shortcut3(out)
            print("第八层relu之前的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)
            out = F.relu(out)
            print("第⑧层relu之后的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)

            out1 = self.conv9(out)
            print("第九层relu之前的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.myReLU5(out1)
            print("第九层relu之后的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.conv10(out1)
            out = out1 + self.shortcut4(out)
            print("第Ⅹ层relu之前的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)
            out = F.relu(out)
            print("第⑩层relu之后的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)

            out1 = self.conv11(out)
            print("第十一层relu之前的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.myReLU6(out1)
            print("第十一层relu之后的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.conv12(out1)
            out = out1 + self.shortcut5(out)
            print("第12层relu之前的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)
            out = F.relu(out)
            print("第12层relu之前的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)

            out1 = self.conv13(out)
            print("第13层relu之前的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.myReLU7(out1)
            print("第13层relu之后的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.conv14(out1)
            out = out1 + self.shortcut6(out)
            print("第14层relu之前的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)
            out = F.relu(out)
            print("第14层relu之后的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)

            out1 = self.conv15(out)
            print("第15层relu之前的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.myReLU8(out1)
            print("第15层relu之后的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.conv16(out1)
            out = out1 + self.shortcut7(out)
            print("第16层relu之前的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)
            out = F.relu(out)
            print("第16层relu之后的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)

            out1 = self.conv17(out)
            print("第17层relu之前的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.myReLU9(out1)
            print("第17层relu之后的数据:", numpy.array2string(out1.cpu().numpy(), separator=',').encode(), file=f)
            out1 = self.conv18(out1)
            out = out1 + self.shortcut8(out)
            print("第18层relu之前的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)
            out = F.relu(out)
            print("第18层relu之后的数据:", numpy.array2string(out.cpu().numpy(), separator=',').encode(), file=f)

            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

class ResNetTest(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetTest, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.myReLU1 = nn.Sequential(
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.myReLU2 = nn.Sequential(
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut0 = nn.Sequential()
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.myReLU3 = nn.Sequential(
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut1 = nn.Sequential()
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.myReLU4 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut2 = nn.Sequential()
        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU5 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut3 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(32)
                )
        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU6 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut4 = nn.Sequential()
        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU7 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut5 = nn.Sequential()
        self.conv13 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU8 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut6 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64)
                )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU9 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut7 = nn.Sequential()
        self.conv17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU10 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut8 = nn.Sequential()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        numpy.set_printoptions(threshold=sys.maxsize)
        out = self.conv0(x)

        out1 = self.conv1(out)
        # out1 = MyReLU.apply(out1)
        # temp_str = numpy.array2string(out1.cpu().numpy(), separator=',').encode()

        out1 = self.myReLU1(out1)
        out1 = self.conv2(out1)
        out = out1 + self.shortcut0(out)
        out = F.relu(out)

        out1 = self.conv3(out)
        out1 = self.myReLU2(out1)
        out1 = self.conv4(out1)
        out = out1 + self.shortcut1(out)
        out = F.relu(out)

        out1 = self.conv5(out)
        out1 = self.myReLU3(out1)
        out1 = self.conv6(out1)
        out = out1 + self.shortcut2(out)
        out = F.relu(out)

        out1 = self.conv7(out)
        out1 = self.myReLU4(out1)
        out1 = self.conv8(out1)
        out = out1 + self.shortcut3(out)
        out = F.relu(out)

        out1 = self.conv9(out)
        out1 = self.myReLU5(out1)
        out1 = self.conv10(out1)
        out = out1 + self.shortcut4(out)
        out = F.relu(out)

        out1 = self.conv11(out)
        out1 = self.myReLU6(out1)
        out1 = self.conv12(out1)
        out = out1 + self.shortcut5(out)
        out = F.relu(out)

        out1 = self.conv13(out)
        out1 = self.myReLU7(out1)
        out1 = self.conv14(out1)
        out = out1 + self.shortcut6(out)
        out = F.relu(out)

        out1 = self.conv15(out)
        out1 = self.myReLU8(out1)
        out1 = self.conv16(out1)
        out = out1 + self.shortcut7(out)
        out = F.relu(out)

        out1 = self.conv17(out)
        out1 = self.myReLU9(out1)
        out1 = self.conv18(out1)
        out = out1 + self.shortcut8(out)
        out = F.relu(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNetRelux8(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetRelux8, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.myReLU1 = nn.Sequential(
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.myReLU2 = nn.Sequential(
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut0 = nn.Sequential()
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.myReLU3 = nn.Sequential(
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut1 = nn.Sequential()
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.myReLU4 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut2 = nn.Sequential()
        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU5 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut3 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(32)
                )
        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU6 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut4 = nn.Sequential()
        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU7 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut5 = nn.Sequential()
        self.conv13 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU8 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut6 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64)
                )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU9 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut7 = nn.Sequential()
        self.conv17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU10 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut8 = nn.Sequential()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        numpy.set_printoptions(threshold=sys.maxsize)
        out = self.conv0(x)

        out1 = self.conv1(out)
        out1 = self.myReLU1(out1)
        # out1 = MyReLUx8.apply(out1)
        out1 = self.conv2(out1)
        out = out1 + self.shortcut0(out)
        out = F.relu(out)
        # out = MyReLUx8.apply(out)

        out1 = self.conv3(out)
        out1 = self.myReLU2(out1)
        # out1 = MyReLUx8.apply(out1)
        out1 = self.conv4(out1)
        out = out1 + self.shortcut1(out)
        out = F.relu(out)
        # out = MyReLUx8.apply(out)

        out1 = self.conv5(out)
        out1 = self.myReLU3(out1)
        # out1 = MyReLUx8.apply(out1)
        out1 = self.conv6(out1)
        out = out1 + self.shortcut2(out)
        out = F.relu(out)
        # out = MyReLUx8.apply(out)

        out1 = self.conv7(out)
        out1 = self.myReLU4(out1)
        # out1 = MyReLUx8.apply(out1)
        out1 = self.conv8(out1)
        out = out1 + self.shortcut3(out)
        out = F.relu(out)
        # out = MyReLUx8.apply(out)

        out1 = self.conv9(out)
        out1 = self.myReLU5(out1)
        # out1 = MyReLUx8.apply(out1)
        out1 = self.conv10(out1)
        out = out1 + self.shortcut4(out)
        out = F.relu(out)
        # out = MyReLUx8.apply(out)

        out1 = self.conv11(out)
        out1 = self.myReLU6(out1)
        # out1 = MyReLUx8.apply(out1)
        out1 = self.conv12(out1)
        out = out1 + self.shortcut5(out)
        out = F.relu(out)
        # out = MyReLUx8.apply(out)

        out1 = self.conv13(out)
        out1 = self.myReLU7(out1)
        # out1 = MyReLUx8.apply(out1)
        out1 = self.conv14(out1)
        out = out1 + self.shortcut6(out)
        out = F.relu(out)
        # out = MyReLUx8.apply(out)

        out1 = self.conv15(out)
        out1 = self.myReLU8(out1)
        # out1 = MyReLUx8.apply(out1)
        out1 = self.conv16(out1)
        out = out1 + self.shortcut7(out)
        out = F.relu(out)
        # out = MyReLUx8.apply(out)

        out1 = self.conv17(out)
        out1 = self.myReLU9(out1)
        # out1 = MyReLUx8.apply(out1)
        out1 = self.conv18(out1)
        out = out1 + self.shortcut8(out)
        # out = F.relu(out)
        out = MyReLUx8.apply(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNetRelux7(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetRelux7, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.myReLU1 = nn.Sequential(
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.myReLU2 = nn.Sequential(
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut0 = nn.Sequential()
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.myReLU3 = nn.Sequential(
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut1 = nn.Sequential()
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.myReLU4 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut2 = nn.Sequential()
        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU5 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut3 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(32)
                )
        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU6 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut4 = nn.Sequential()
        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU7 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut5 = nn.Sequential()
        self.conv13 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU8 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut6 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64)
                )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU9 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut7 = nn.Sequential()
        self.conv17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU10 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut8 = nn.Sequential()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        numpy.set_printoptions(threshold=sys.maxsize)
        out = self.conv0(x)

        out1 = self.conv1(out)
        # out1 = self.myReLU1(out1)
        out1 = MyReLUx7.apply(out1)
        out1 = self.conv2(out1)
        out = out1 + self.shortcut0(out)
        out = F.relu(out)
        # out = MyReLUx7.apply(out)

        out1 = self.conv3(out)
        # out1 = self.myReLU2(out1)
        out1 = MyReLUx7.apply(out1)
        out1 = self.conv4(out1)
        out = out1 + self.shortcut1(out)
        out = F.relu(out)
        # out = MyReLUx7.apply(out)

        out1 = self.conv5(out)
        # out1 = self.myReLU3(out1)
        out1 = MyReLUx7.apply(out1)
        out1 = self.conv6(out1)
        out = out1 + self.shortcut2(out)
        out = F.relu(out)
        # out = MyReLUx7.apply(out)

        out1 = self.conv7(out)
        # out1 = self.myReLU4(out1)
        out1 = MyReLUx7.apply(out1)
        out1 = self.conv8(out1)
        out = out1 + self.shortcut3(out)
        out = F.relu(out)
        # out = MyReLUx7.apply(out)

        out1 = self.conv9(out)
        # out1 = self.myReLU5(out1)
        out1 = MyReLUx7.apply(out1)
        out1 = self.conv10(out1)
        out = out1 + self.shortcut4(out)
        out = F.relu(out)
        # out = MyReLUx7.apply(out)

        out1 = self.conv11(out)
        # out1 = self.myReLU6(out1)
        out1 = MyReLUx7.apply(out1)
        out1 = self.conv12(out1)
        out = out1 + self.shortcut5(out)
        out = F.relu(out)
        # out = MyReLUx7.apply(out)

        out1 = self.conv13(out)
        # out1 = self.myReLU7(out1)
        out1 = MyReLUx7.apply(out1)
        out1 = self.conv14(out1)
        out = out1 + self.shortcut6(out)
        # out = F.relu(out)
        out = MyReLUx7.apply(out)

        out1 = self.conv15(out)
        # out1 = self.myReLU8(out1)
        out1 = MyReLUx7.apply(out1)
        out1 = self.conv16(out1)
        out = out1 + self.shortcut7(out)
        out = F.relu(out)
        # out = MyReLUx7.apply(out)

        out1 = self.conv17(out)
        # out1 = self.myReLU9(out1)
        out1 = MyReLUx7.apply(out1)
        out1 = self.conv18(out1)
        out = out1 + self.shortcut8(out)
        out = F.relu(out)
        # out = MyReLUx7.apply(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNetRelux6(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetRelux6, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.myReLU1 = nn.Sequential(
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.myReLU2 = nn.Sequential(
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut0 = nn.Sequential()
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.myReLU3 = nn.Sequential(
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut1 = nn.Sequential()
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.myReLU4 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.shortcut2 = nn.Sequential()
        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU5 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut3 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(32)
                )
        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU6 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut4 = nn.Sequential()
        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.myReLU7 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.shortcut5 = nn.Sequential()
        self.conv13 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU8 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut6 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64)
                )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU9 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut7 = nn.Sequential()
        self.conv17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.myReLU10 = nn.Sequential(
            nn.ReLU(),
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut8 = nn.Sequential()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        numpy.set_printoptions(threshold=sys.maxsize)
        out = self.conv0(x)

        out1 = self.conv1(out)
        # out1 = self.myReLU1(out1)
        out1 = MyReLUx6.apply(out1)
        out1 = self.conv2(out1)
        out = out1 + self.shortcut0(out)
        out = F.relu(out)
        # out = MyReLUx6.apply(out)

        out1 = self.conv3(out)
        # out1 = self.myReLU2(out1)
        out1 = MyReLUx6.apply(out1)
        out1 = self.conv4(out1)
        out = out1 + self.shortcut1(out)
        out = F.relu(out)
        # out = MyReLUx6.apply(out)

        out1 = self.conv5(out)
        # out1 = self.myReLU3(out1)
        out1 = MyReLUx6.apply(out1)
        out1 = self.conv6(out1)
        out = out1 + self.shortcut2(out)
        out = F.relu(out)
        # out = MyReLUx6.apply(out)

        out1 = self.conv7(out)
        # out1 = self.myReLU4(out1)
        out1 = MyReLUx6.apply(out1)
        out1 = self.conv8(out1)
        out = out1 + self.shortcut3(out)
        # out = F.relu(out)
        out = MyReLUx6.apply(out)

        out1 = self.conv9(out)
        # out1 = self.myReLU5(out1)
        out1 = MyReLUx6.apply(out1)
        out1 = self.conv10(out1)
        out = out1 + self.shortcut4(out)
        # out = F.relu(out)
        out = MyReLUx6.apply(out)

        out1 = self.conv11(out)
        # out1 = self.myReLU6(out1)
        out1 = MyReLUx6.apply(out1)
        out1 = self.conv12(out1)
        out = out1 + self.shortcut5(out)
        out = F.relu(out)
        # out = MyReLUx6.apply(out)

        out1 = self.conv13(out)
        # out1 = self.myReLU7(out1)
        out1 = MyReLUx6.apply(out1)
        out1 = self.conv14(out1)
        out = out1 + self.shortcut6(out)
        # out = F.relu(out)
        out = MyReLUx6.apply(out)

        out1 = self.conv15(out)
        # out1 = self.myReLU8(out1)
        out1 = MyReLUx6.apply(out1)
        out1 = self.conv16(out1)
        out = out1 + self.shortcut7(out)
        out = F.relu(out)
        # out = MyReLUx6.apply(out)

        out1 = self.conv17(out)
        # out1 = self.myReLU9(out1)
        out1 = MyReLUx6.apply(out1)
        out1 = self.conv18(out1)
        out = out1 + self.shortcut8(out)
        out = F.relu(out)
        # out = MyReLUx6.apply(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet20():
    return ResNet(ResidualBlock)

def ResNet20TestSavefile():
    return ResNetTestSavefile()

# 测试暴露所有的网络结构
def ResNet20Test():
    return ResNetTest()

# 测试relu用x8替换
def ResNet20relux8():
    return ResNetRelux8()

# 测试relu用x7替换
def ResNet20relux7():
    return ResNetRelux7()

# 测试relu用x6替换
def ResNet20relux6():
    return ResNetRelux6()
