import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("analysing_environmental_issues.csv")
print(data['work_shift'])

# приводим данные к правильному виду
data['DateTime'] = pd.to_datetime(data['DateTime'])

data['X'] = range(1, len(data) + 1)  # добавляем дополнительную переменную для посторения графиков

# фильтруем столбцы с колличественным типом данных, не включая 'stage_4_output_danger_gas'
numeric_columns = data.select_dtypes(include=[np.number]).columns.to_list()
numeric_columns.remove('stage_4_output_danger_gas')

# построение графиков до изменения
for column in numeric_columns:
    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    plt.suptitle(column, fontsize=16, y=1.01)

    ax[1].scatter(data['X'], data[column], label=column, color='blue')

    sns.boxplot(data[column], ax=ax[0])
    plt.savefig(fr'graphics\before\before_changes_{column}')


# функция для определения верхних и нижних границ
def calculate_boundaries(column, wight_up, wight_bottom):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - wight_bottom * IQR
    upper_bound = Q3 + wight_up * IQR
    return lower_bound, upper_bound


# функция для обработки выбросов с заменой на значение границ
def handle_outliers(column, wight_up, wight_bottom):
    lower_bound, upper_bound = calculate_boundaries(column, wight_up, wight_bottom)
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])


# функция для обработки выбросов с заменой на среднее значение
def handle_outliers_mean(column, wight_up, wight_bottom):
    lower_bound, upper_bound = calculate_boundaries(column, wight_up, wight_bottom)
    data[column] = np.where(data[column] < lower_bound, data[column].mean(), data[column])
    data[column] = np.where(data[column] > upper_bound, data[column].mean(), data[column])


# проверка и замена выбросов с помощью введённых выше функций
for column in numeric_columns:
    handle_outliers(column, 3, 3)

# определяем столбцы и способ работы с ними
inter15 = ['stage_1_output_konv_avd', "stage_2_output_bottom_pressure", 'stage_2_output_top_pressure_at_end',
           "stage_2_output_top_vacuum",
           'stage_3_input_pressure', "stage_4_input_steam", 'stage_4_output_product', 'stage_4_output_dry_residue_avg']
inter15_UP = ['stage_2_output_top_temp', 'stage_3_output_temp_top']
inter15_down = ['stage_2_output_bottom_temp', 'stage_2_output_top_pressure', 'stage_3_output_temp_hum_steam',
                'stage_4_input_overheated_steam']
inter20_down = ['stage_2_output_bottom_temp_hum_steam', 'stage_2_output_bottom_vacuum', 'stage_3_input_soft_water',
                'stage_4_input_polymer']

# изменение с помощью интерквартильного размаха
for column in numeric_columns:
    handle_outliers(column, 3, 3)

for column in inter15:
    handle_outliers_mean(column, 1.5, 1.5)

for column in inter15_UP:
    handle_outliers_mean(column, 1.5, 3)

for column in inter15_down:
    handle_outliers_mean(column, 3, 1.5)

for column in inter20_down:
    handle_outliers(column, 2, 2)

# изменение с помощью интерполяции
# с лимитом замены нескольких значений подряд = 1
for column in numeric_columns:
    data[column] = data[column].interpolate(limit=1)

data = data.dropna(subset=numeric_columns)

# построение графиков после изменения
for column in numeric_columns:
    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    plt.suptitle(column, fontsize=16, y=1.01)

    ax[1].scatter(data['X'], data[column], label=column, color='blue')

    sns.boxplot(data[column], ax=ax[0])
    plt.savefig(fr'graphics\after\after_changes_{column}')

# заменяем значения в столбце 'work_shift' на целые
data['work_shift'] = data['work_shift'].astype('int')

missing_data = data.isnull().sum()
data = data.drop(columns='X')

data['amount_input_danger_gas'] = data['stage_4_output_danger_gas'].apply(lambda x:
                                                                          'низкий' if x < 0.1 else 'средний'
                                                                          if 0.1 <= x < 0.4 else 'высокий')
# для удобства создаём новый измененный файл
data.to_csv('output.csv')

print(data.info())
# print("Пропуски:\n", missing_data)
