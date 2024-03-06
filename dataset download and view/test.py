# import PIL.Image
import torchvision

fashion_mnist_test = torchvision.datasets.FashionMNIST(root="../data_fashion", train=True, transform=None, download=True)
mnist_test = torchvision.datasets.MNIST(root="../data", train=False, transform=None, download=True)

mnist_test[0][0].show() # 图片，这个类型是 PIL.Image.Image，这个类型有 .show() 方法
print(mnist_test[0][1]) # 标注

fashion_mnist_test[0][0].show()
print(fashion_mnist_test[0][1])