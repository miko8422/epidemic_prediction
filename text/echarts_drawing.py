# -*- coding: utf-8 -*-
# @Project_Name: epidemic_pridiction
# @Time    : 2021/5/8 15:55
# @Author  : miko
# @Site    : 
# @File    : echarts_drawing.py
# @Software: PyCharm
from BPNN import *
net=torch.load('bp.pkl')

net.eval()  # 将模型改为预测模式
with torch.no_grad():
    # for i in range(246):
    im = Variable(Test_Data_Input[0]) # torch中训练需要将其封装即Variable，此处封装像素即784
    label = Variable(torch.from_numpy(np.array([0.0020])))    # 此处为标签
    out = net(label)  # 经网络输出的结果
    # if (i ==0 ):
    #     Predict_Out.append(out*10)
    # elif(i<17):
    #     Predict_Out.append((out*10))
    # else:
    Predict_Out.append(F.relu(out))

    loss = criterion(out, label)  # 得到误差
        # 记录误差
    eval_loss += loss.item()
        # 记录准确率
    eval_losses.append(eval_loss)
    predicted_cases = preprocessing.MinMaxScaler().fit(ALL_EXCEL_DAT_0124_S).inverse_transform(
        np.expand_dims(Predict_Out, axis=0)
    ).flatten()
    print('epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(e, train_loss,eval_loss ))

import matplotlib.pyplot as plt

plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)
plt.savefig(fname="03.png")
plt.show()
movie_name = np.array([i for i in range(246)])
first_day = np.array(ALL_EXCEL_DAT.iloc[:,33])


first_weekend = np.array(abs(predicted_cases), dtype = np.float16)

# 先得到movie_name长度, 再得到下标组成列表
x = len(movie_name)
total_width, n = 2, 2     # 有多少个类型，只需更改n即可
width = total_width / n
x = x - (total_width - width) / 2
print(first_weekend)
# print(first_weekend[0:10])
# print(first_day)        #y轴值
# print(first_weekend)    #y轴值
# print(movie_name)       #x轴值



import pyecharts.options as opts
from pyecharts.charts import Bar3D
data=[]
for i in range(len(movie_name)):
    data.append([i,0,first_day[i]])
    data.append([i,1,first_weekend[i]])
(
    Bar3D(init_opts=opts.InitOpts(width="800px", height="600px"))
    .add(
        series_name="",
        data=data,
        xaxis3d_opts=opts.Axis3DOpts(type_="category", data=movie_name),
        yaxis3d_opts=opts.Axis3DOpts(type_="category"),
        zaxis3d_opts=opts.Axis3DOpts(type_="value"),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            max_=200,
            range_color=[
                "#313695",
                "#4575b4",
                "#74add1",
                "#abd9e9",
                "#e0f3f8",
                "#ffffbf",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
                "#a50026",
            ],
        )
    )
    .render("bar3d_punch_card.html")
)

# plt.bar(movie_name, first_day,  width=0.5, label='label1',color='deepskyblue')
# plt.bar(movie_name + 0.5, first_weekend, width=0.5, label='label2',color='red')
# # 底部汉字移动到两个柱状条中间(本来汉字是在左边蓝色柱状条下面, 向右移动0.1)  。。。0.5吧
# plt.xticks()
#
# plt.savefig(fname="04.png")
# plt.show()

