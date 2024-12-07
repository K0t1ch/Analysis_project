import pandas as pd
import numpy as np

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
