from torch.utils.data import DataLoader
from torchvision import datasets,transforms

def read_mnist(batchsize,data_dir):
    # 定义转换操作，将图像转换为Tensor，并归一化到[0-1]范围内
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
    )
    
    # 下载并加载MNIST数据集
    train_set = datasets.MNIST(root=data_dir, 
                            train=True, 
                            download=True, 
                            transform=transform)
    
    test_set = datasets.MNIST(root=data_dir, 
                            train=False, 
                            download=True, 
                            transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(dataset=train_set, 
                            batch_size=batchsize, 
                            shuffle=True)
    
    test_loader = DataLoader(dataset=test_set, 
                            batch_size=batchsize, 
                            shuffle=True)
    
    return train_loader, test_loader