import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from matplotlib import pyplot
from torch.distributions import normal

# 这里模型形如 y = w_0*x_0 + w_1*x_1 + b

true_w = torch.tensor([3, -1.2])
true_b = 2.1
def datamaker(source_w, source_b, want_num):
    # 这里就是直接按照我们的假设来生成数据
    _data_X = torch.normal(0, 1, (want_num, len(source_w)))
    _data_y = torch.mv(_data_X, source_w) + source_b
    _data_y += torch.normal(0, 0.01, _data_y.shape)
    return _data_X, _data_y

data_X, data_y = datamaker(true_w, true_b, 100)


'''
# 用 matplotlib 分别可视化 x_0 和 x_1 对于 y 的线性关系

plt.title("just a stupid title")
plt.xlim(-10,10)
plt.xlabel("X")
plt.ylim(-10, 10)
plt.ylabel("Y")

plt.scatter(data_X[:, 0].numpy(), data_y.numpy()) # 试了，这里不用先 .detach() 就可以直接转成 np.ndarry
plt.show()

plt.scatter(data_X[:, 1].numpy(), data_y.numpy())
plt.show()
'''

batch_size = 10
def dataiter(batch_size, X, y) : # yeild 的意思是这个函数里的局部变量在 yeild 一次后会暂停保留当前状态
    all_size = len(X)            # 下次调用才会运行到下一次 yeild
                                 # yeild 相当于普通函数里的 return
    index_temp = list(range(all_size))
    random.shuffle(index_temp) # 生成乱序标号，等于是随机抽取样本了
    for i in range(0, all_size, batch_size) :
        batch_index = torch.tensor(index_temp[i : min(all_size, i + batch_size)])
        yield X[batch_index, :], y[batch_index]
'''
for X, y in dataiter(batch_size, data_X, data_y) :
    print(X)
    print(y)
    print("-----------")
'''

def linear_model(X, w, b) :
    return torch.mv(X, w) + b

def loss(y_hat, data_y) :
    return (y_hat - data_y) ** 2 / 2 / len(data_y)

# def batch_upd()

# 为模型旋转初始参数
model_w = normal.Normal(0., 1.).sample([2]) # torch.normal 的采样只能生成矩阵或更高维的，我不知道怎么生成向量
model_b = torch.zeros((1))

model_w.requires_grad = True
model_b.requires_grad = True


learn_rate = 0.1
train_epochs = 5

epochlist = []
losslist = []
w0list = []
w1list = []

for epoch in range(train_epochs) :
    for batch_X, batch_y in dataiter(batch_size, data_X, data_y) :
        batch_loss = loss(linear_model(batch_X, model_w, model_b), batch_y)
        batch_loss.sum().backward()

        with torch.no_grad() :
            model_w -= model_w.grad * learn_rate
            model_w.grad.zero_()
            model_b -= model_b.grad * learn_rate
            model_b.grad.zero_()

    with torch.no_grad() :
        now_model_loss = loss(linear_model(data_X, model_w, model_b), data_y)
        print("epoch : ", epoch + 1, ", loss : ", now_model_loss.sum())

        epochlist.append(epoch)
        losslist.append(now_model_loss.sum())
        w0list.append(float(model_w[0]))
        w1list.append(float(model_w[1]))

print("model_w : ", model_w, ", model_b : ", model_b)

# plt.cla()
plt.title("a title")
plt.grid() # 背景加上网格


plt.plot(epochlist, losslist, label="loss")
plt.plot(epochlist, w0list, label="w0")
plt.plot(epochlist, w1list, label="w1")
plt.legend() # 加上这个才能显示 label

plt.show()