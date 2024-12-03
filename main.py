import pandas as pd
import numpy as np

df = pd.read_csv('analysing_environmental_issues.csv')
df['work_shift'].ffill()

# заменяем все пропуски средним значением

for i in ['stage_2_output_bottom_temp', 'stage_2_output_bottom_temp_hum_steam',
          'stage_2_output_top_temp',
          'stage_3_output_temp_hum_steam', 'stage_3_output_temp_top', 'stage_4_output_danger_gas',
          'stage_2_input_water_sum', 'stage_2_output_bottom_pressure', 'stage_2_output_bottom_vacuum',
          'stage_2_output_top_pressure', 'stage_2_output_top_pressure_at_end', 'stage_2_output_top_vacuum',
          'stage_3_input_pressure', 'stage_3_input_soft_water', 'stage_3_input_steam', 'stage_4_input_overheated_steam',
          'stage_4_input_polymer', 'stage_4_input_steam', 'stage_4_input_water',
          'stage_4_output_dry_residue_avg',
          'stage_4_output_product'
          ]:
    df[i] = df[i].fillna(df[i].mean())

# ---------------------------------------------------------------------
# проанализировав данные, можно увидеть
# слишком высокие значения.
# заменить их выбросы следует медианой.

# столбцы с супервыбросами - stage_2_input_water_sum, stage_2_output_bottom_pressure,
# stage_2_output_bottom_vacuum, stage_2_output_top_pressure, stage_2_output_top_pressure_at_end,
# stage_3_input_soft_water, stage_3_input_steam, stage_4_input_steam, stage_4_input_water

for i in ['stage_2_input_water_sum', 'stage_2_output_bottom_pressure', 'stage_2_output_bottom_vacuum',
          'stage_2_output_top_pressure', 'stage_2_output_top_pressure_at_end',
          'stage_3_input_soft_water', 'stage_3_input_steam',
          'stage_4_input_steam', 'stage_4_input_water']:

    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1

    low = Q1 - 3 * IQR
    big = Q3 + 3 * IQR

    for j in range(len(df)):
        if df.loc[j, i] < low or df.loc[j, i] > big:
            df.loc[j, i] = df[i].median()

# ---------------------------------------------------------------------

# столбцы с выбросами - stage_2_output_bottom_temp,
# stage_2_output_bottom_temp_hum_steam, stage_2_output_top_temp, stage_2_output_top_vacuum,
# stage_3_input_pressure, stage_3_output_temp_hum_steam, stage_3_output_temp_top, stage_4_input_overheated_steam,
# stage_4_input_polymer, stage_4_output_danger_gas, stage_4_output_dry_residue_avg, stage_4_output_product

for i in ['stage_2_output_bottom_vacuum',
          'stage_3_input_pressure', 'stage_4_input_overheated_steam',
          'stage_4_input_polymer', 'stage_4_output_danger_gas',
          'stage_4_output_dry_residue_avg',
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

# аналогичная ситуация и с праметрами температур, но в этом случае, изменение значения выбросов стоит проводить с
# помощью среднего, а не медианы, ведь в этом случае значение температуры вполне
# может отражать реальную структура данных. Аналогичная ситуация и с количеством выбросов вредных газов.

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

# для более оптимального анализа зависимости выбросов вредных газов, я создал отдельный столбец с качественными данными,
# который определяет степень выбросов

df['amount_input_danger_gas'] = df['stage_4_output_danger_gas'].apply(lambda x:
                                                                      'низкий' if x < 0.1 else 'средний'
                                                                      if 0.1 <= x < 0.4 else 'высокий')

print(df.describe())
