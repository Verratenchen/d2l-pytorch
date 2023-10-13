# 初始化方案的选择在神经网络学习中起着举足轻重的作用，它对保持数值稳定性至关重要。
# 我们选择哪个函数以及如何初始化参数可以决定优化算法收敛的速度有多快。


# 4.8.1
# 梯度爆炸（gradient exploding）：参数更新过大，破坏了模型的稳定收敛
# 梯度消失（gradient vanishing）：参数更新过小，在每次更新时几乎不会移动，导致模型无法学习

# 梯度消失
import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))

d2l.plt.show()


# 梯度爆炸
M = torch.normal(0, 1, size=(4, 4))
print('一个矩阵\n', M)
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))

print('乘以 100 个矩阵后\n', M)

# 其参数化所固有的对称性
# 虽然小批量随机梯度下降不会打破这种对称性，但暂退法正则化可以。


# 4.8.2
# 解决（或至少减轻）上述问题的一种方法是进行参数初始化，优化期间的注意和适当的正则化也可以进一步提高稳定性。

# 默认初始化
# 默认的随机初始化方法

# Xavier 初始化
# 通常，Xavier 初始化从均值为零，方差 sigma^2 = 2/(n_in + n_out) 的高斯分布中采样权重。
# 也可以将其改为选择从均匀分布中抽取权重时的方差。

