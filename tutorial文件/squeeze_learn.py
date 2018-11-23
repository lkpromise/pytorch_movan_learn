# 添加维度（unsqueeze) 与去掉维度 (squeeze)


import  torch
# 制作数据，维度为1*3
a = torch.randn(2,3,3)
# 打印a及a 的规模
print(a)
print(a.view(a.size(0),-1))
# 去掉a中维度为1的维度
b = torch.squeeze(a)
# b 从1行三列，变为3列，虽然形式没变，但已经不再是一个矩阵，而是个数组
print(b)
print(b.shape)
# 在b的第1个位置添加一个维度
c = torch.unsqueeze(b,1)
print(c)
print(c.shape)
d = torch.squeeze(c)
print(d)
print(d.shape)