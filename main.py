import pandas as pd
import numpy as np

df = pd.read_csv('analysing_environmental_issues.csv')
df.fillna(df['stage_1_output_konv_avd'].mean(), inplace=True)
df['work_shift'].ffill()

# столбцы с супервыбросами - stage_2_input_water_sum, stage_2_output_bottom_pressure,
# stage_2_output_bottom_vacuum, stage_2_output_top_pressure, stage_2_output_top_pressure_at_end,
# stage_3_input_soft_water, stage_3_input_steam, stage_4_input_steam, stage_4_input_water

# столбцы с выбросами - stage_2_output_bottom_temp,
# stage_2_output_bottom_temp_hum_steam, stage_2_output_top_temp, stage_2_output_top_vacuum,
# stage_3_input_pressure, stage_3_output_temp_hum_steam, stage_3_output_temp_top, stage_4_input_overheated_steam,
# stage_4_input_polymer, stage_4_output_danger_gas, stage_4_output_dry_residue_avg, stage_4_output_product

# проанализировав данные стобца 'stage_2_input_water_sum', можно увидеть
# слишком высокие значения: 'max       233.370000', что сильно превышает значение 'mean       86.674616'.
# заменить эти выбросы следует медианой.

for i in ['stage_2_input_water_sum', 'stage_2_output_bottom_pressure', 'stage_2_output_bottom_vacuum',
          'stage_2_output_top_pressure', 'stage_2_output_top_pressure_at_end', 'stage_2_output_top_vacuum',
          'stage_3_input_pressure', 'stage_3_input_soft_water', 'stage_3_input_steam', 'stage_4_input_overheated_steam',
          'stage_4_input_polymer', 'stage_4_input_steam', 'stage_4_input_water',
          'stage_4_output_danger_gas', 'stage_4_output_dry_residue_avg',
          'stage_4_output_product']:

    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    big = Q3 + 1.5 * IQR

    for j in range(len(df)):
        if df.loc[j, i] < low or df.loc[j, i] > big:
            df.loc[j, i] = df[i].median()

# ---------------------------------------------------------------------

# аналогичная ситуация и с праметрами температур, но в этом случае, изменение значения выбросов стоит проводить с помощью
# среднего, а не медианы, ведь в этом случае значение температуры вполне может отражать реальную структура данных

for i in ['stage_2_output_bottom_temp', 'stage_2_output_bottom_temp_hum_steam',
          'stage_2_output_top_temp',
          'stage_3_output_temp_hum_steam', 'stage_3_output_temp_top']:

    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    big = Q3 + 1.5 * IQR

    for j in range(len(df)):
        if df.loc[j, i] < low or df.loc[j, i] > big:
            df.loc[j, i] = df[i].mean()

print(df.info())
