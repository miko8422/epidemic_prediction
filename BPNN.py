#BPNN神经网络模型训练


#from pyecharts import options as opts
#from pyecharts.charts import Bar#柱状图所导入的包
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import seaborn as sns               #数据可视化
from matplotlib.pylab import rcParams   #绘图（折线图）
from pandas.plotting import register_matplotlib_converters
from sklearn import preprocessing

#设置图配置
#美化展示效果
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()


ALL_EXCEL_DAT = pd.read_excel('AI 2020 04 data01.xlsx',sheet_name = 1).dropna() # 取出数据，清除空位
EXCEL_DAT_IN_0124_S  = np.array(ALL_EXCEL_DAT.iloc[:,0:15])#数据里面的迁入人数
EXCEL_DAT_OUT_0124_S = np.array(ALL_EXCEL_DAT.iloc[:,15:30])#数据里面的迁出人数
ALL_EXCEL_DAT_0124_S = np.expand_dims(ALL_EXCEL_DAT.iloc[:,33], axis=1)     #扩展了三列
# print(ALL_EXCEL_DAT_0124_S)
#对数据进行归一化
EXCEL_DAT_IN_0124  = preprocessing.MinMaxScaler().fit_transform(EXCEL_DAT_IN_0124_S)
EXCEL_DAT_OUT_0124 = preprocessing.MinMaxScaler().fit_transform(EXCEL_DAT_OUT_0124_S)
ALL_EXCEL_DAT_0124 = preprocessing.MinMaxScaler().fit_transform(ALL_EXCEL_DAT_0124_S)

Data_1_2 = np.zeros(shape=(246,30))#因为excel里面的数据原因，初始化设定好为246*30的零矩阵
for i in range(246):#将迁入人数和迁出人数相邻交叉合在一起
    for j in range(15):
        Data_1_2[i][ 2*j ] = EXCEL_DAT_IN_0124[i][j]
        Data_1_2[i][2*j+1] = EXCEL_DAT_OUT_0124[i][j]
#我们把前200行的数据作为训练集，剩下的作为测试集
Train_Data_Input  = torch.from_numpy(Data_1_2[0:100]).float()
Train_Data_Output = torch.from_numpy(ALL_EXCEL_DAT_0124[0:200]).float()


Test_Data_Input   = torch.from_numpy(Data_1_2).float()
Test_Data_Output  = torch.from_numpy(ALL_EXCEL_DAT_0124).float()


# 使用 Sequential 定义 4 层神经网络
net = nn.Sequential(
    nn.Linear(30 , 512),
    nn.LeakyReLU(),
    nn.Linear(512, 200),
    nn.Dropout(0.1),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ELU(),
    nn.Linear(100, 1),
)
# net
# 定义 loss 函数
criterion = nn.MSELoss(reduction='mean') # 定义

optimizer = torch.optim.ASGD(net.parameters(), 0.01)  # 使用随机梯度下降，学习率 0.1    #学习率越高，损失函数的损失也就越多？
# 开始训练，定义空列表进而存储后边训练得到的数据
losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(100):
    train_loss = 0
    net.train()
    for i in range(100):
        im = Variable(Train_Data_Input[i])
        label = Variable(Train_Data_Output[i])
         # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
    losses.append(train_loss)
# print(len(losses))
los_list = []
for i in losses:
    los_list.append(i)
    # 在测试集上检验效果
eval_loss = 0
eval_acc = 0
Predict_Out=[]

torch.save(net, 'bp.pkl')
# net=torch.load()
# """
net.eval()  # 将模型改为预测模式
with torch.no_grad():
    # for i in range(246):
    data=[79, 72, 76, 65, 62, 67, 64, 61, 58, 54, 55, 67, 65, 71, 72, 83, 96, 103, 107, 114, 114, 106, 100, 115, 112, 111, 97, 96, 105, 104]
    data.reverse()
    im = Variable(torch.from_numpy(np.array(data)).float())        #torch中训练需要将其封装即Variable，此处封装像素即784
    # label = Variable(Test_Data_Output[0])    # 此处为标签
    out = net(im)  # 经网络输出的结果
    print(out)
    # if (i ==0 ):
    #     Predict_Out.append(out*10)
    # elif(i<17):
    #     Predict_Out.append((out*10))
    # else:
    Predict_Out.append(F.relu(out))

    # loss = criterion(out, label)  # 得到误差
        # 记录误差
    # eval_loss += loss.item()
        # 记录准确率
    # eval_losses.append(eval_loss)
    predicted_cases = preprocessing.MinMaxScaler().fit(np.array([[i] for i in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 10, 13, 27, 36, 41, 48, 52, 58, 61, 69, 69, 75, 85, 85, 85, 78, 64, 55, 49, 41, 38, 31, 27, 21, 87, 85, 85, 79, 72, 76, 65, 62, 67, 64, 61, 58, 54, 55, 67, 65, 71, 72, 83, 96, 103, 107, 114, 114, 106, 100, 115, 112, 111, 97, 96, 105, 104]])).inverse_transform(
        np.expand_dims(Predict_Out, axis=0)
    ).flatten()
    print(predicted_cases)

    print('epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(e, train_loss,eval_loss ))

import matplotlib.pyplot as plt

plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)

plt.show()
movie_name = np.array([i for i in range(246)])
first_day = np.array(ALL_EXCEL_DAT.iloc[:,33])


first_weekend = np.array(abs(predicted_cases), dtype = np.float16)

# 先得到movie_name长度, 再得到下标组成列表
x = len(movie_name)
total_width, n = 2, 2     # 有多少个类型，只需更改n即可
width = total_width / n
x = x - (total_width - width) / 2
print(first_weekend[0:10])
plt.bar(movie_name, first_day,  width=0.5, label='label1',color='deepskyblue')
plt.bar(movie_name + 0.5, first_weekend, width=0.5, label='label2',color='red')
# 底部汉字移动到两个柱状条中间(本来汉字是在左边蓝色柱状条下面, 向右移动0.1)  。。。0.5吧
plt.xticks()
plt.show()


# """


