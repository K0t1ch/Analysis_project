import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, zscore

df = pd.read_csv('analysing_environmental_issues.csv', sep=',', decimal='.')
df['work_shift'] = df['work_shift'].ffill()
df['work_shift'] = df['work_shift'].astype('int')

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

stages = [['stage_2_input_water_sum', 'stage_2_output_bottom_pressure', 'stage_2_output_bottom_temp',
           'stage_2_output_bottom_temp_hum_steam', 'stage_2_output_bottom_vacuum', 'stage_2_output_top_pressure',
           'stage_2_output_top_pressure_at_end', 'stage_2_output_top_temp', 'stage_2_output_top_vacuum'],
          ['stage_3_input_pressure', 'stage_3_input_soft_water', 'stage_3_input_steam',
           'stage_3_output_temp_hum_steam', 'stage_3_output_temp_top'],
          ['stage_4_input_overheated_steam', 'stage_4_input_polymer', 'stage_4_input_steam', 'stage_4_input_water',
           'stage_4_output_danger_gas', 'stage_4_output_dry_residue_avg', 'stage_4_output_product']]

for i in range(5):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    sns.histplot(data=sp_df[i][stages[0]], ax=axes[0][1], kde=False, bins=30).set(ylabel='', xlabel='stage_2')
    sns.histplot(data=sp_df[i][stages[1]], ax=axes[1][0], kde=False, bins=30).set(ylabel='', xlabel='stage_3')
    sns.histplot(data=sp_df[i][stages[2]], ax=axes[1][1], kde=False, bins=30).set(ylabel='', xlabel='stage_4')
    sns.boxplot(ax=axes[0][0], data=sp_df[i]['stage_1_output_konv_avd']).set(ylabel='', xlabel='stage_1')

    fig.suptitle(f"Гистограмма и ящик с усами для стадий производства до изменений день {i + 1}")

    plt.savefig(rf'graphics\before\before_changes_{i + 1}')

for dataf in sp_df:
    dataf = dataf.dropna()
    if len(dataf) > 3:
        for i in names:
            if i != 'amount_input_danger_gas':
                if i != 'stage_4_output_danger_gas':
                    if dataf[i].max() - dataf[i].min() > 0:
                        p_value = shapiro(dataf[i])[1]

                        if p_value >= 0.05:
                            for j in range(len(df)):
                                df.loc[j, i] = df[i].mean()

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

                                    df.loc[j, i] = df[i].median()

                            elif (abs(zscore(dataf[i])) >= 3.29).any():
                                Q1 = dataf[i].quantile(0.25)
                                Q3 = dataf[i].quantile(0.75)

                                IQR = Q3 - Q1
                                lower_bound = Q1 - 6 * IQR
                                upper_bound = Q3 + 6 * IQR

                                for j in range(len(df)):
                                    if df.loc[j, i] < lower_bound or df.loc[j, i] > upper_bound:
                                        df.loc[j, i] = df[i].median()

                                    df.loc[j, i] = df[i].median()

for i in range(5):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    sns.histplot(data=sp_df[i][stages[0]], ax=axes[0][1], kde=False, bins=30).set(ylabel='', xlabel='stage_2')
    sns.histplot(data=sp_df[i][stages[1]], ax=axes[1][0], kde=False, bins=30).set(ylabel='', xlabel='stage_3')
    sns.histplot(data=sp_df[i][stages[2]], ax=axes[1][1], kde=False, bins=30).set(ylabel='', xlabel='stage_4')
    sns.boxplot(ax=axes[0][0], data=sp_df[i]['stage_1_output_konv_avd']).set(ylabel='', xlabel='stage_1')

    fig.suptitle(f"Гистограмма и ящик с усами для стадий производства после изменений день {i + 1}")

    plt.savefig(rf'graphics\after\after_changes_{i + 1}')

print(df.info())
