#LSTM神经网络训练

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pandas.plotting import register_matplotlib_converters
from torch import nn
import datetime
import xlwt, xlrd
import xlutils.copy
def dateRange(beginDate, endDate):
    dates = []
    dt = datetime.datetime.strptime(beginDate, "%Y-%m-%d")
    date = beginDate[:]
    while date <= endDate:
        dates.append(date)
        dt = dt + datetime.timedelta(1)
        date = dt.strftime("%Y-%m-%d")
    return dates
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
#读取Excel表里面所有数据并且删除带有空值的行
#print(ALL_EXCEL_DAT.head())
#print(ALL_EXCEL_DAT.isnull().T.sum())
ALL_EXCEL_DAT = pd.read_excel('AI 2020 04 data01.xlsx',sheet_name = 1).dropna()

EXCEL_DAT_IN_0124_S  = np.array(ALL_EXCEL_DAT.iloc[:,0:15])
EXCEL_DAT_OUT_0124_S = np.array(ALL_EXCEL_DAT.iloc[:,15:30])
excel_all_in_shape_s = np.array(ALL_EXCEL_DAT.iloc[0:9,0:14]).transpose()

excel_all_in_shape = preprocessing.MinMaxScaler().fit_transform(excel_all_in_shape_s)
EXCEL_DAT_IN_0124  = preprocessing.MinMaxScaler().fit_transform(EXCEL_DAT_IN_0124_S)
EXCEL_DAT_OUT_0124 = preprocessing.MinMaxScaler().fit_transform(EXCEL_DAT_OUT_0124_S)

test_data_size = 100
train_data = EXCEL_DAT_IN_0124[:test_data_size,]
test_data = EXCEL_DAT_IN_0124_S[-test_data_size:,]
# 数据处理，升高维度，给每个单位增加维度  感觉并未加维度，只是转置了一下
def FunctionUp(train_data):
    train_data_list,train_data_lists=[i for i in range(len(train_data[0]))],[]
    for i in range(len(train_data)):
        for j in range(len(train_data[0])):
            train_data_list[j]=[train_data[i][j]]
        train_data_lists.append(train_data_list)
        train_data_list=[i for i in range(len(train_data[0]))]
    return np.array(train_data_lists)

#切片处理
def create_sequences(In_Data, seq_length):
    xl,xs = [],[]
    yl,ys = [],[]
    for j in range(len(In_Data)):
        for i in range(len(In_Data[0]) - seq_length - 1):
            x = In_Data[j][i:(i + seq_length)]
            y = In_Data[j][i + seq_length]
            xs.append(x)
            ys.append(y)
        xl.append(xs)
        yl.append(ys)
        xs, ys = [], []
    return np.array(xl), np.array(yl)
seq_length = 5
X_train, y_train = create_sequences(FunctionUp(train_data), seq_length)
X_test, y_test = create_sequences(FunctionUp(test_data), seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()


# %%

class CoronaVirusPredictor(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(CoronaVirusPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )
    #前向传播
    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = \
            lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

def train_model(
        model,
        train_data,
        train_labels,
        test_data=None,
        test_labels=None
):
    data = xlrd.open_workbook("excel_data/excel_data.xls")  # 读入表格
    ws = xlutils.copy.copy(data)  # 复制之前表里存在的数据
    table = ws.get_sheet(0)

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        for i in range(len(X_train)):
            model.reset_hidden_state()

            Train_Data_Input_Pred = model(X_train[i])
            loss = loss_fn(Train_Data_Input_Pred.float(), y_train[i])

            if(t == num_epochs-1):

                for c in range(len(Train_Data_Input_Pred)):
                    table.write(i, c, label="%.2f" % Train_Data_Input_Pred[c])  # 因为单元格从0开始算，所以row不需要加一

        if test_data is not None:
            with torch.no_grad():
                Test_Data__Pred = model(X_train[i])
                test_loss = loss_fn(Test_Data__Pred.float(), y_train[i])




            test_hist[t] = test_loss.item()
            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')

        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')
        train_hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    ws.save("excel_data/excel_data.xls")  # 保存的有旧数据和新数据

    return model.eval(), train_hist, test_hist



# %%
model = CoronaVirusPredictor(
    n_features = 1,
    n_hidden = 512,
    seq_len = seq_length,
    n_layers = 2
)
model, train_hist, test_hist = train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
)

los_list = []
for i in train_hist:
    los_list.append(i)

plt.plot([i for i in range(100)],train_hist, color='blue',label="Training loss",linewidth=2.0)
plt.plot([i for i in range(100)],test_hist, label="Test loss",linewidth=3.0)
plt.ylim((0, 0.2))
plt.legend()
plt.savefig(fname="01.png")
plt.show()
# %%
# predicting daily cases
# %%

'''
with torch.no_grad():
    all_preds=[i for i in range(len(X_test))]
    for i in range(len(X_test)):
        test_seq = X_test[i][:1]
        preds = []
        for _ in range(len(X_test[0])):
            y_test_pred = model(test_seq)
            pred = torch.flatten(y_test_pred).item()
            preds.append(pred)
            new_seq = test_seq.numpy().flatten()
            new_seq = np.append(new_seq, [pred])
            new_seq = new_seq[1:]
            test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
        all_preds[i]=preds
y_test_np=np.zeros((len(y_test),len(y_test[0])))
for i in range(len(y_test)):
    y_test_np[i]=np.array(np.expand_dims(y_test[i].flatten().numpy(), axis=0))

true_cases = preprocessing.MinMaxScaler().fit(excel_all_in_shape_s).inverse_transform(
    np.array(y_test_np)
)
predicted_cases = preprocessing.MinMaxScaler().fit(excel_all_in_shape_s).inverse_transform(
    np.array(all_preds)
)
print(all_preds)
print(predicted_cases.shape)

plt.plot([i for i in range(len(y_test_np))], y_test_np, color='red', linewidth=2.0, linestyle='--',label='y_test_np')
plt.plot([i for i in range(len(all_preds))], predicted_cases, color='blue', linewidth=3.0, linestyle='-.',label='all_preds')
plt.legend()
plt.show()
'''
y_test_np=np.zeros((len(y_test),len(y_test[0])))
for i in range(len(y_test)):
    y_test_np[i]=np.array(np.expand_dims(y_test[i].flatten().numpy(), axis=0))

true_cases = preprocessing.MinMaxScaler().fit(excel_all_in_shape_s).inverse_transform(
    np.array(y_test_np)
)

DAYS_TO_PREDICT = 9
test_line = 5

print(y_train[0])


# %%
with torch.no_grad():
    test_seq = X_test[0][test_line:test_line+1] #取第test_line维度数据做最终预测
    preds = []
    for _ in range(DAYS_TO_PREDICT):
        y_test_pred = model(test_seq)
        print(y_test_pred)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
# %%

predicted_cases = preprocessing.MinMaxScaler().fit(excel_all_in_shape_s).inverse_transform(
    np.expand_dims(preds, axis=0)
).flatten()
def writeadd():

    data = xlrd.open_workbook("excel_data/excel_data.xls")#读入表格
    ws = xlutils.copy.copy(data) #复制之前表里存在的数据
    table = ws.get_sheet(0)
    for i in range(len(predicted_cases)):
        table.write(i, 1, label="%.2f"%predicted_cases[i])  # 因为单元格从0开始算，所以row不需要加一
    ws.save("excel_data/excel_data.xls")  #保存的有旧数据和新数据
writeadd()

plt.plot([date for date in dateRange('2020-01-10', '2020-01-18')],excel_all_in_shape_s[test_line], color='red', linewidth=2.0, linestyle='--',label='y_test_np')
plt.plot([date for date in dateRange('2020-01-10', '2020-01-18')],predicted_cases, label='Predicted Daily   ')
plt.legend()
plt.savefig(fname="02.png")
plt.show()

