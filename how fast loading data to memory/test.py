# i7-12650H
# 1.61 sec (worknum = 0)
# windows 下必须在整个程序前加 if __name__ == '__main__':
# 才能开多线程，不然会报错
# 1.37 sec (worknum = 4)
# 可以看到数据集较小的时候多线程也没那么大收益
import torchvision
from torchvision import transforms
from torch.utils import data
import time
import numpy as np

if __name__ == '__main__':
    class Timer:
        """记录多次运行时间"""
        def __init__(self):
            self.times = []
            self.start()

        def start(self):
            """启动计时器"""
            self.tik = time.time()

        def stop(self):
            """停止计时器并将时间记录在列表中"""
            self.times.append(time.time() - self.tik)
            return self.times[-1]

        def avg(self):
            """返回平均时间"""
            return sum(self.times) / len(self.times)

        def sum(self):
            """返回时间总和"""
            return sum(self.times)

        def cumsum(self):
            """返回累计时间"""
            return np.array(self.times).cumsum().tolist()

    fashion_mnist_train = torchvision.datasets.FashionMNIST(root="../data_fashion", train=True, transform=transforms.ToTensor(), download=True)
    fashion_mnist_test = torchvision.datasets.FashionMNIST(root="../data_fashion", train=False, transform=transforms.ToTensor(), download=True)

    batch_size = 256
    fashion_mnist_train_iter = data.DataLoader(dataset=fashion_mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

    timer = Timer()

    timer.start()
    for x,y in fashion_mnist_train_iter :
        continue

    print(f'{timer.stop():.2f} sec')

# print(type(fashion_mnist_train[0][0]))