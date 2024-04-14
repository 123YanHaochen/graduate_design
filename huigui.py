import torch
import torch.nn as nn
import numpy as np



x_values=[i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train=x_train.reshape(-1,1)

y_values=[3*i for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train=y_train.reshape(-1,1)


print(x_train,y_train)
'''
线性回归模型
'''
class LinearRegressionModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        out = self.linear(x)
        return out
model = LinearRegressionModel(1,1)



'''
指定参数和损失函数
'''
epochs = 6000
lr=0.01
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
criterion = nn.MSELoss()

for epoch in range(epochs):
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss=criterion(outputs,labels)  #计算模型输出与实际标签之间的损失
    loss.backward()    #这一步将梯度信息存储在模型的参数中，以便进行参数更新
    optimizer.step()    #用优化器，根据梯度更新模型参数。这是梯度下降的步骤，通过学习率控制参数的更新步长。
    if epoch%50==0:
        print('epoch {},loss {}'.format(epoch,loss.item()))
# 查看网络参数
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Value: {param.data}")
torch.save(model.state_dict(), 'model.pkl')
input_data = torch.tensor([[3.5]], dtype=torch.float32)  # 将要预测的数据转换为张量

# 关闭梯度计算
with torch.no_grad():
    # 将输入数据传递给模型进行预测
    output = model(input_data)

# 获取预测结果
prediction = output.item()
print("Prediction:", prediction)