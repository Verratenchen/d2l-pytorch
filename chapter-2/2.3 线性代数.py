# 2.3.1
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x **y)


# 2.3.2
x = torch.arange(4)
print(x)
print(x[3])

print(len(x))
print(x.shape)


# 2.3.3
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)


# 2.3.4
X = torch.arange(24).reshape(2, 3, 4)
print(X)


# 2.3.5
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将 A 的一个副本分配给 B
print(A)
print(A + B)
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a * X).shape)


# 2.3.6
x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())
print(A.shape, A.sum())

A_sum_axis0 = A.sum(axis=0)
print(A)
print(A_sum_axis0)
print(A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A_sum_axis1.shape)

print(A.sum(axis=[0, 1]))

print(A.mean())
print(A.sum() / A.numel())

print(A.mean(axis=0))
print(A.sum(axis=0) / A.shape[0])

sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(A / sum_A)

print(A.cumsum(axis=0))


# 2.3.7
y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))

print(torch.sum(x * y))


# 2.3.8
print(A.shape, x.shape, torch.mv(A, x))


# 2.3.9
B = torch.ones(4, 3)
print(torch.mm(A, B))


# 2.3.10
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

print(torch.abs(u).sum())

print(torch.norm(torch.ones((4, 9))))
