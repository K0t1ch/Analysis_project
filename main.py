import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, zscore

df = pd.read_csv('analysing_environmental_issues.csv', sep=',', decimal='.')

# данные в столбце 'work_shift' заменены на ближайшие,
# т.к. здесь наблюдается плавное измение данных, в рамках значений 1 и 2
df['work_shift'] = df['work_shift'].ffill()

# тип данных изменён на целочисленный
df['work_shift'] = df['work_shift'].astype('int')

sp_df = []
index_old = 0
df1 = ''

# при помощи цикла происходит разбивение общего датафрейма на отдельные, с интервалом измерения в один час,
# для более правильной обработки данных
while index_old < 4398:
    for i in range(len(df) - 1):
        try:
            if int(df['DateTime'][i + 1].split()[1].split(':')[0]) - int(df['DateTime'][i].split()[1].split(':')[0]) != 1:
                df1 = df[index_old:i + 1]
                sp_df.append(df1)
                df1 = ''
                index_old = i + 1

# производится обработка подразумеваемой ошибки при проверке интервала между данными
        except Exception:
            pass

# ---------------------------------------------------------------------

# создается категориальный столбец с уровнем выброса газа
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

# для анализа значений производится построение 5-ти график по 5-ти первым таблицам соответственно
for i in range(5):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    sns.histplot(data=sp_df[i][stages[0]], ax=axes[0][1], kde=False, bins=30).set(ylabel='', xlabel='stage_2')
    sns.histplot(data=sp_df[i][stages[1]], ax=axes[1][0], kde=False, bins=30).set(ylabel='', xlabel='stage_3')
    sns.histplot(data=sp_df[i][stages[2]], ax=axes[1][1], kde=False, bins=30).set(ylabel='', xlabel='stage_4')
    sns.boxplot(ax=axes[0][0], data=sp_df[i]['stage_1_output_konv_avd']).set(ylabel='', xlabel='stage_1')

    fig.suptitle(f"Гистограмма и ящик с усами для стадий производства до изменений день {i + 1}")
    plt.savefig(rf'graphics\before\before_changes_{i + 1}')

# -----ОБРОБОТКА ВЫБРОСОВ И ЗАМЕНА ПРОПУСКОВ-----
for dataf in sp_df:
    dataf = dataf.dropna()
    if len(dataf) > 3:      # размер каждого датайфрейма из списка проверяется, чтобы определить уместность проверки
        for i in names:
            if i != 'amount_input_danger_gas':

                # данные в категориальном столбце выбросов газа не изменяются,т.к. в тип данных в столбце - str,
                # не подразумевающие выбросов

                if i != 'stage_4_output_danger_gas':

                    # аналогично данные в колличественном столбце не изменяются,
                    # так как в данном столбце слишком много пропусков

                    # чтобы метод Шапиро применялся корректно, происходит проверка диапазона данных, который показывает
                    # одиниковы ли значения в столбце, т.к. при одинаковых данных тест может выдать неверное значение
                    if dataf[i].max() - dataf[i].min() > 0:
                        p_value = shapiro(dataf[i])[1]

                        if p_value >= 0.05:
                            # если данные распределены нормально - пропуски заменяются средним значением,
                            # так как подобные данные вполне могут отражать реальную структуру,
                            # а также могут повлиять на последующий анализ
                            for j in range(len(df)):
                                df.loc[j, i] = df[i].mean()

                        else:

                            # в ином случае, при помощи Z-оценки, определяется степень выброса
                            # чтобы избежать ошибки - программа работает с выбросами,
                            # если хотя бы одно значение отклоняется от подразумеваемого для выброса
                            # или супервыброса соответсвенно

                            if (abs(zscore(dataf[i])) >= 1.96).any():
                                # для замены выбросов используеься интерквартильный размах с множителем 3

                                Q1 = dataf[i].quantile(0.25)
                                Q3 = dataf[i].quantile(0.75)

                                IQR = Q3 - Q1
                                lower_bound = Q1 - 3 * IQR
                                upper_bound = Q3 + 3 * IQR

                                for j in range(len(df)):
                                    if df.loc[j, i] < lower_bound or df.loc[j, i] > upper_bound:
                                        df.loc[j, i] = df[i].mean()

                                    df.loc[j, i] = df[i].median()
                                    # пропущенные значения заменяются медианой,
                                    # так как она менее восприимчива к выбросам

                            elif (abs(zscore(dataf[i])) >= 3.29).any():
                                # для работы с аномалиями также используется интреквартильный размах, но с множителем 6

                                Q1 = dataf[i].quantile(0.25)
                                Q3 = dataf[i].quantile(0.75)

                                IQR = Q3 - Q1
                                lower_bound = Q1 - 6 * IQR
                                upper_bound = Q3 + 6 * IQR

                                for j in range(len(df)):
                                    if df.loc[j, i] < lower_bound or df.loc[j, i] > upper_bound:
                                        df.loc[j, i] = df[i].median()

                                    df.loc[j, i] = df[i].median()

# чтобы определить измененные значения,
# производится построение новых 5-ти графиков всё по тем же первым 5-ти датафреймам из списка

for i in range(5):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    sns.histplot(data=sp_df[i][stages[0]], ax=axes[0][1], kde=False, bins=30).set(ylabel='', xlabel='stage_2')
    sns.histplot(data=sp_df[i][stages[1]], ax=axes[1][0], kde=False, bins=30).set(ylabel='', xlabel='stage_3')
    sns.histplot(data=sp_df[i][stages[2]], ax=axes[1][1], kde=False, bins=30).set(ylabel='', xlabel='stage_4')
    sns.boxplot(ax=axes[0][0], data=sp_df[i]['stage_1_output_konv_avd']).set(ylabel='', xlabel='stage_1')

    fig.suptitle(f"Гистограмма и ящик с усами для стадий производства после изменений день {i + 1}")
    plt.savefig(rf'graphics\after\after_changes_{i + 1}')

print(df.info())
