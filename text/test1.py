import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from pandas.plotting import register_matplotlib_converters
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()


a =[11618,
2141,
1897,
885,
471,
915,
838,
553,
610,
635
]
b=[9095.1,
888.9,
1184.7,
359.6,
149,
320.,
248.6,
367.8,
190.6,
188.4
]
c =[3857.14,
578.4,
682.8,
228.0,
175.9,
233.4,
280.8,
230.4,
1821,
1558
]
d = [9580.64,
1275,
2050,
964,
114.06,
78.94,
567,
231.5,
254.8,
231.8
]

plt.figure()
lenth = [i for i in range(len(b))]
print(lenth)
plt.title('ContrastFigure')

plt.bar(lenth, b, label="LSTM",color='blue')
plt.bar(lenth, c, label="BPNN", color='green')
plt.bar(lenth, d, label="LSTM+BPNN" ,color='red' )

plt.legend()
#plt.savefig("avt_comper1.svg", format="svg")
plt.show()
