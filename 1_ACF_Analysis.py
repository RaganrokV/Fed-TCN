import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.tsa.api as smt
import pandas as pd
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)


#%%
# 数据读取与预处理
data_csv = pd.read_csv('EEMD-Pollutants\Roadside data-SH.csv')
pd.set_option('display.max_columns', 20)
# divide 4 stations
DF_station = data_csv[data_csv['站名'].isin(['浦东东方路交通站'])]
DF_station = DF_station.fillna(DF_station.interpolate())

JY_station = data_csv[data_csv['站名'].isin(['静安延安西路交通站'])]
JY_station = JY_station.fillna(JY_station.interpolate())

JG_station = data_csv[data_csv['站名'].isin(['静安共和新路交通站'])]
JG_station = JG_station.fillna(JG_station.interpolate())

CX_station = data_csv[data_csv['站名'].isin(['徐汇漕溪路交通站'])]
CX_station = CX_station.fillna(CX_station.interpolate())
#%%
fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")
acf1 = smt.stattools.acf(CX_station['NO(μg/m³)'].iloc[:24*365], nlags=7*24-1)
plt.plot(acf1)
# plt.axvline(x=24)
plt.ylabel('ACF')
plt.xticks([0,1*24,2*24,3*24,4*24,5*24,6*24,7*24],
           [' 1-lag ','24-lag','48-lag','72-lag','96-lag',
            '120-lag','144-lag','168-lag',],rotation=30)
# plt.savefig("EEMD-Pollutants/Figs/ACF.svg", dpi=600)
plt.show()
#%%
fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")
acf1 = smt.stattools.acf(CX_station['NO(μg/m³)'].iloc[:24*365], nlags=2*24-1)
plt.plot(acf1)
# plt.axvline(x=24)
plt.ylabel('ACF')
plt.xticks([0,1*24,2*24],
           [' 0-lag ','24-lag','48-lag',],rotation=0)
# plt.savefig("EEMD-Pollutants/Figs/ACF.svg", dpi=600)
plt.show()