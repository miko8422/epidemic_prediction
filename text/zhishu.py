import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from pandas.plotting import register_matplotlib_converters
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()
from statsmodels.tsa.holtwinters import ExponentialSmoothing
ALL_EXCEL_DAT = pd.read_excel('AI 2020 04 data01.xlsx',sheet_name = 1).dropna()

EXCEL_DAT_IN_0124_S  = np.array(ALL_EXCEL_DAT.iloc[:,0:15])
EXCEL_DAT_OUT_0124_S = np.array(ALL_EXCEL_DAT.iloc[:,15:30])
x3 = (EXCEL_DAT_IN_0124_S[1])[0:15]
#x3 = np.linspace(0, 4 * np.pi, 100)
y3 = pd.Series( x3)
print(y3)
ets3 = ExponentialSmoothing(y3, trend='add', seasonal='add', seasonal_periods=5)
r3 = ets3.fit()
pred3 = r3.predict(start=len(y3[0:10]), end=len(y3[0:10]) + len(y3[0:10])//2-1)
pd.DataFrame({
    'TrueData': y3,
    'FittingData': r3.fittedvalues,
    'PredData': pred3,

}).plot(legend=True,linewidth=5.0)
plt.savefig('1.svg', format="svg")
plt.show()
