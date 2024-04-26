from torchvision import datasets,transforms
from torch.utils.data import DataLoader   #导入下载通道

def read_cifar100(batchsize,data_dir):
    transform_train = transforms.Compose(
        [transforms.Resize(256),           #transforms.Scale(256)
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    transform_test = transforms.Compose(
        [transforms.Resize(256),         #transforms.Scale(256)
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
    # 训练数据集
    train_data = datasets.CIFAR100(root =data_dir,
                            train = True,
                            transform = transform_train,
                            download = True)
    
    # 测试数据集
    test_data = datasets.CIFAR100(root =data_dir,
                            train = False,
                            transform = transform_test,
                            download = True)

    data_loader_train = DataLoader(dataset=train_data,
                                batch_size=batchsize,
                                shuffle=True,    #打乱数据
                                pin_memory=True)   #内存充足时，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False
                                #drop_last=True)   #处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
    data_loader_test = DataLoader(dataset=test_data,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True)
    return data_loader_train,data_loader_test
