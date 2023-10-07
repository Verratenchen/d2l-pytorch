# 2.1.1
import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

X = x.reshape(3,4)
print(X)
X_1 = x.reshape(-1,4)
X_2 = x.reshape(3,-1)
print(X_1)
print(X_2)

print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))
print(torch.randn(3, 4))
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))


# 2.1.2
# 按元素计算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)
print(torch.exp(x))

# 连结（concatenate）
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X)
print(Y)
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))

print(X == Y)

print(X.sum())


# 2.1.3
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b)


# 2.1.4
print(X[-1])
print(X[1:3])
X[1, 2] = 9
print(X)

X[0:2, :] = 12
print(X)


# 2.1.5
before = id(Y)
Y = Y + X
print(id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

before = id(X)
X += Y
print(id(X) == before)


# 2.1.6
A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))