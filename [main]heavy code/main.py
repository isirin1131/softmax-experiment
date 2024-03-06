'''
数据量比较小，开多线程也快不了多少，这里就不开了
'''
import torch
# import PIL.Image
import torchvision
from torchvision import transforms
from torch.utils import data

fashion_mnist_train = torchvision.datasets.FashionMNIST(root="../data_fashion", train=True, transform=transforms.ToTensor(), download=True)
fashion_mnist_test = torchvision.datasets.FashionMNIST(root="../data_fashion", train=False, transform=transforms.ToTensor(), download=True)

# fashion_mnist_train[0][0].show()
# print(len(fashion_mnist_train), len(fashion_mnist_test))
# print(fashion_mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# print(get_fashion_mnist_labels([0, 1, 2, 3]))

batch_size = 256
train_iter = data.DataLoader(dataset=fashion_mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)

# 测一下一个 batch 长什么样
# print(next(iter(train_iter))[1][1])
# x, y = next(iter(train_iter))
# print(x.shape, y.shape)
# print(x[0][0].reshape((-1)).shape) # torch.Size([784])

len_input = 784
len_output = 10

model_W = torch.normal(mean=0, std=0.01, size=(len_input, len_output), requires_grad=True)
model_B = torch.zeros(size=(len_output,), requires_grad=True)


''' 测试一下向量除以标量张量的广播机制
o = torch.ones(size=(10,))
partition = torch.ones(size=(10,))
print(partition.shape, o.shape)
partition = partition.sum()
print(partition.shape)
print(o / partition)
'''

def softmax(o) :
    exp_o = torch.exp(o)
    partition = exp_o.sum(dim=1, keepdim=True)
    return exp_o / partition

''' 测试一下 softmax 运算（旧版本）
ttt = softmax(torch.tensor([-1, -2, -3, 1, 2, 3, 4, 5, 6]))
print(ttt, ttt.sum())

output = 
tensor([5.7735e-04, 2.1240e-04, 7.8136e-05, 4.2661e-03, 1.1596e-02, 3.1522e-02,
        8.5686e-02, 2.3292e-01, 6.3314e-01]) tensor(1.)
'''

def net(X, W, b) : # 每行是每个 X 的预测值 o
    return softmax(torch.mm(X, W) + b)

def crossentropy_loss(hat_y, y) :
    return - torch.log(hat_y[range(len(hat_y)), y])

''' 对 argmax 的实验
Test = torch.tensor([[0.1, 0.2, 0.3], [3, 2, 1], [10, 100, 0.01]])
Test_2 = Test.argmax(dim=1, keepdim=True)
Test_2 = torch.exp(Test_2)
print(torch.mm(Test, Test_2))
'''

