import torch
import torch.utils.data as Data
# 制作假数据
torch.manual_seed(1)  # reproducible
x = torch.linspace(1, 10, 10)  # x data
#print(x.size())
y = torch.linspace(10, 1, 10)  # y data
# 定义每次处理的数据大小
BATCH_SIZE = 5
# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(x,y)

# 把dataset放入dataloader
loader = Data.DataLoader(dataset=torch_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=2, )
for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        # 假设这里是训练的地方

        print('Epoch: ', epoch, '| Step: ', step, '| batch x :', batch_x.numpy(), '| batch y: ', batch_y.numpy())
