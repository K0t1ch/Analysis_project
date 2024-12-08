import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, zscore

df = pd.read_csv('analysing_environmental_issues.csv', sep=',', decimal='.')
df['work_shift'] = df['work_shift'].ffill()
df['work_shift'] = df['work_shift'].astype('int')

# df = df[df['покупательская_активность'] == input()].reset_index(drop=True)
# fig, axes = plt.subplots(1, 2, figsize=(9, 3))
#
# sns.histplot(x=df['выручка_от_клиента_текущий_месяц'], ax=axes[0], kde=True, bins=30).set(ylabel='', xlabel='')
# sns.boxplot(ax=axes[1], y=df['выручка_от_клиента_текущий_месяц']).set(ylabel='', xlabel='')
#
# fig.suptitle("Гистограмма и ящик с усами для количественных данных")
#
# plt.savefig('target_4_6.png')

sp_df = []
index_old = 0
df1 = ''

while index_old < 4398:
    for i in range(len(df) - 1):
        try:
            if int(df['DateTime'][i + 1].split()[1].split(':')[0]) - int(df['DateTime'][i].split()[1].split(':')[0]) != 1:
                df1 = df[index_old:i + 1]
                sp_df.append(df1)
                df1 = ''
                index_old = i + 1

        except Exception:
            print(f"{i} {df['DateTime'][i]}")

# ---------------------------------------------------------------------

df['amount_input_danger_gas'] = df['stage_4_output_danger_gas'].apply(lambda x:
                                                                      'низкий' if x < 0.05 else 'средний'
                                                                      if 0.05 <= x < 0.16 else 'высокий' if x >= 0.16
                                                                      else x)

# ---------------------------------------------------------------------

names = df.columns[1:-2]

for dataf in sp_df:
    dataf = dataf.dropna()
    if len(dataf) > 3:
        for i in names:
            if i != 'amount_input_danger_gas':
                if i != 'stage_4_output_danger_gas':
                    if dataf[i].max() - dataf[i].min() > 0:
                        p_value = shapiro(dataf[i])[1]

                        if p_value >= 0.05:
                            pass

                        else:
                            if (abs(zscore(dataf[i])) >= 1.96).any():
                                Q1 = dataf[i].quantile(0.25)
                                Q3 = dataf[i].quantile(0.75)

                                IQR = Q3 - Q1
                                lower_bound = Q1 - 3 * IQR
                                upper_bound = Q3 + 3 * IQR

                                for j in range(len(df)):
                                    if df.loc[j, i] < lower_bound or df.loc[j, i] > upper_bound:
                                        df.loc[j, i] = df[i].mean()

                            elif (abs(zscore(dataf[i])) >= 3.29).any():
                                Q1 = dataf[i].quantile(0.25)
                                Q3 = dataf[i].quantile(0.75)

                                IQR = Q3 - Q1
                                lower_bound = Q1 - 6 * IQR
                                upper_bound = Q3 + 6 * IQR

                                for j in range(len(df)):
                                    if df.loc[j, i] < lower_bound or df.loc[j, i] > upper_bound:
                                        df.loc[j, i] = df[i].median()

print(df.info())
