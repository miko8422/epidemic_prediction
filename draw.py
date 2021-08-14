#数据展示


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from pandas.plotting import register_matplotlib_converters
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()
ALL_EXCEL_DAT = pd.read_excel('text/AI 2020 04 data01.xlsx',sheet_name = 1).dropna()

EXCEL_DAT_IN_0124_S  = np.array(ALL_EXCEL_DAT.iloc[:,0:15])
EXCEL_DAT_OUT_0124_S = np.array(ALL_EXCEL_DAT.iloc[:,15:30])
#a.py
print(EXCEL_DAT_IN_0124_S[100:])

c = EXCEL_DAT_IN_0124_S[100:].flatten()
v = EXCEL_DAT_OUT_0124_S[100:].flatten()
for i in EXCEL_DAT_IN_0124_S[100:]:
    for j in i:
        if j >5:
            print(j)
    #s 表示散点的大小，形如 shape (n, )
    #label 表示显示在图例中的标注
    #alpha 是 RGBA 颜色的透明分量
    #edgecolors 指定三点圆周的颜色
plt.scatter(v,c,c="green",label="Low",alpha=0.6,edgecolors='white')

a = EXCEL_DAT_IN_0124_S[35:100].flatten()
b = EXCEL_DAT_OUT_0124_S[35:100].flatten()
plt.scatter(b,a,c="blue",label="Medium",alpha=0.6,edgecolors='white')
x = EXCEL_DAT_IN_0124_S[0:35].flatten()
y = EXCEL_DAT_OUT_0124_S[0:35].flatten()

    #s 表示散点的大小，形如 shape (n, )
    #label 表示显示在图例中的标注
    #alpha 是 RGBA 颜色的透明分量
    #edgecolors 指定三点圆周的颜色
plt.scatter(y,x,c="red",label="High",alpha=0.6,edgecolors='white')
plt.title('RiskDistribution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('05.png')
plt.show()