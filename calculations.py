import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr, spearmanr, chi2_contingency, shapiro, mannwhitneyu

data = pd.read_csv('output.csv', sep=',', decimal='.')

# Категоризация опасного газа
data['danger_category'] = pd.cut(data['stage_4_output_danger_gas'],
                                 bins=[0, 0.05, 0.16, 1],
                                 labels=['низкая', 'средняя', 'высокая'])

# ----Сводная таблица изменения входных параметров по дням----

# Суммарные показатели по дням

daily_summary = data.groupby(data['DateTime'].str[:10]).agg({
    'stage_2_input_water_sum': 'sum',
    'stage_3_input_pressure': 'sum',
    'stage_3_input_soft_water': 'sum',
    'stage_3_input_steam': 'sum',
    'stage_4_input_overheated_steam': 'sum',
    'stage_4_input_polymer': 'sum',
    'stage_4_input_steam': 'sum',
    'stage_4_input_water': 'sum'
}).reset_index()

print("Сводная таблица по дням:")
print(daily_summary)

daily_summary_melted = daily_summary.melt(id_vars='DateTime',
                                          value_vars=[
                                              'stage_2_input_water_sum',
                                              'stage_3_input_pressure',
                                              'stage_3_input_soft_water',
                                              'stage_3_input_steam',
                                              'stage_4_input_overheated_steam',
                                              'stage_4_input_polymer',
                                              'stage_4_input_steam',
                                              'stage_4_input_water'
                                          ],
                                          var_name='Input Parameter',
                                          value_name='Value')

# Визуализация всех параметров
plt.figure(figsize=(20, 8))
sns.lineplot(data=daily_summary_melted, x='DateTime', y='Value', hue='Input Parameter')
plt.xticks(ticks=daily_summary['DateTime'][::5], rotation=45)
plt.title('Суммарные входные параметры по дням')
plt.xlabel('Дата')
plt.ylabel('Сумма входных параметров')
plt.tight_layout()
plt.show()

# ----Сводная таблица по месяцам и категория опасности----
# Группировка по месяцам
data['month'] = pd.to_datetime(data['DateTime']).dt.to_period('M')

# Категория опасности, наиболее частая в месяцах
monthly_danger = data.groupby('month')['danger_category'].agg(lambda x: x.mode()[0]).reset_index()

print("Наиболее частая категория опасности по месяцам:")
print(monthly_danger)

# Визуализация
plt.figure(figsize=(12, 10))
sns.countplot(data=data, x='month', hue='danger_category', palette='coolwarm')
plt.title("Частота категорий опасности по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Частота")
plt.xticks(rotation=45)
plt.show()

# ----Средние значения параметров для каждой категории----
# Группировка по категориям опасности
category_means = data.groupby('danger_category').mean(numeric_only=True)
print("Средние значения параметров для каждой категории:")
print(category_means)

# Визуализация
category_means.T.plot(kind='bar', figsize=(15, 8), title="Средние значения параметров по категориям опасности")
plt.xlabel("Параметры")
plt.ylabel("Средние значения")
plt.legend(title="Категория")
plt.show()

# ----Корреляционный анализ----
# Корреляционная матрица
correlation_matrix = data.corr(numeric_only=True, method='spearman')
# во время дальнейших работ с данными было выявлено, что данные распределены ненормально; в связи с этим используется метод спирпена

# Визуализация корреляционной матрицы
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Корреляционная матрица')
plt.tight_layout()
plt.show()

# Вывод корреляций для интересующих столбцов
correlation_danger_gas = correlation_matrix['stage_4_output_danger_gas']
correlation_dry_residue_avg = correlation_matrix['stage_4_output_dry_residue_avg']
correlation_product = correlation_matrix['stage_4_output_product']

print("Корреляции для 'stage_4_output_danger_gas':")
print(correlation_danger_gas)

print("\nКорреляции для 'stage_4_output_dry_residue_avg':")
print(correlation_dry_residue_avg)

print("\nКорреляции для 'stage_4_output_product':")
print(correlation_product)

# ----Проверка гипотез----

# Гипотеза 1: Различия в доле опасного газа от смены
# Таблица сопряженности
contingency_table = pd.crosstab(data['work_shift'], data['danger_category'])

# Хи- квадрат тест
chi2, p, dof, expected = chi2_contingency(contingency_table)

print('Гипотеза 1: Различия в доле опасного газа от смены')
print("Хи-квадрат:", chi2)
print("p-value:", p)

if p < 0.05:
    print("Есть статистически значимые различия в категориях опасности между сменами.")
else:
    print("Нет статистически значимых различий в категориях опасности между сменами.")
print('\n')

# Гипотеза 2: Лучшая смена для управления температурой
# Анализ температуры верха (stage_3_output_temp_top)
shift_1 = data[data['work_shift'] == 1]['stage_3_output_temp_top']
shift_2 = data[data['work_shift'] == 2]['stage_3_output_temp_top']

# Проверим нормальность распределения для каждой смены с помощью теста Шапиро-Уилка
shapiro_shift_1 = shapiro(shift_1)
shapiro_shift_2 = shapiro(shift_2)

# print("Тест Шапиро-Уилка для смены 1:", shapiro_shift_1)
# print("Тест Шапиро-Уилка для смены 2:", shapiro_shift_2)
print('Гипотеза 2: Лучшая смена для управления температурой')
if shapiro_shift_1.pvalue > 0.05 and shapiro_shift_2.pvalue > 0.05:
    t_stat, p_value = ttest_ind(shift_1, shift_2)
    print("t-статистика:", t_stat)
    print("p-значение:", p_value)
else:
    mann_whitney_stat, p_value = mannwhitneyu(shift_1, shift_2)
    print("Статистика Манна-Уитни:", mann_whitney_stat)
    print("p-value Манна-Уитни:", p_value)

    # Выводим результат гипотезы
if p_value < 0.05:
    print("Мы отвергаем нулевую гипотезу: управление температурой между сменами различается.")
else:
    print("Мы не отвергаем нулевую гипотезу: управление температурой между сменами не различается.")
print('\n')

# Гипотеза 3: Конверсия мономера и суммарная вода
konv = data['stage_1_output_konv_avd']
water = data['stage_2_input_water_sum']
shapiro_konv = shapiro(konv)
shapiro_water = shapiro(water)

# print("Тест Шапиро-Уилка:")
# print(f"Процент конверсии сырья: {shapiro_konv}")
# print(f"Суммарная вода: {shapiro_water}")

print("Гипотеза 3: Конверсия мономера на 1 этапе влияет на количество подаваемой суммарной воды:")
# Если данные нормальны, используем корреляцию Пирсона
if shapiro_konv.pvalue > 0.05 and shapiro_water.pvalue > 0.05:
    corr, p_value = pearsonr(konv, water)
    print(f"Корреляция Пирсона между процентом конверсии и суммарной водой: {corr}, p-value: {p_value}")
else:
    corr, p_value = spearmanr(konv, water)
    print(f"Корреляция Спирмена между процентом конверсии и суммарной водой: {corr}, p-value: {p_value}")

# Выводим результаты
if p_value < 0.05:
    print("Конверсия мономера на 1 этапе влияет на количество подаваемой суммарной воды")
else:
    print("Конверсия мономера на 1 этапе не влияет на количество подаваемой суммарной воды")
print('\n')

# Гипотеза 4: Продукт и опасный газ
# Удаляем строки с пропусками в stage_4_output_danger_gas и stage_4_output_product
data_clean = data.dropna(subset=['stage_4_output_danger_gas', 'stage_4_output_product'])

# Данные для анализа
danger_gas = data_clean['stage_4_output_danger_gas']
output_product = data_clean['stage_4_output_product']

# Проверка нормальности данных
shapiro_danger_gas = shapiro(danger_gas)
shapiro_output_product = shapiro(output_product)

# print("Тест Шапиро-Уилка:")
# print(f"Доля опасного газа: p-value = {shapiro_danger_gas.pvalue}")
# print(f"Количество выходного продукта: p-value = {shapiro_output_product.pvalue}")


print("Гипотеза 4: Количество выходного количества продукта связано с долей опасного газа")
if shapiro_danger_gas.pvalue > 0.05 and shapiro_output_product.pvalue > 0.05:
    # Если данные нормальны, используем корреляцию Пирсона
    correlation, p_value = pearsonr(danger_gas, output_product)
    print("Корреляция Пирсона:")
    print(f"Коэффициент корреляции: {correlation}")
    print(f"p-value: {p_value}")
else:
    # Если данные ненормальны, используем корреляцию Спирмена
    correlation, p_value = spearmanr(danger_gas, output_product)
    print("Корреляция Спирмена:")
    print(f"Коэффициент корреляции: {correlation}")
    print(f"p-value: {p_value}")

if p_value < 0.05:
    print("Мы отвергаем нулевую гипотезу: количество выходного продукта связано с долей опасного газа.")
else:
    print("Мы не отвергаем нулевую гипотезу: количество выходного продукта не связано с долей опасного газа.")

# Гипотеза 5: Влияние входного пара на 3 стадии на долю опасного газа
# Разделим данные по категориям опасности
low_danger_steam = data[data['danger_category'] == 'низкая']['stage_3_input_steam']
medium_danger_steam = data[data['danger_category'] == 'средняя']['stage_3_input_steam']
high_danger_steam = data[data['danger_category'] == 'высокая']['stage_3_input_steam']

# Проверим нормальность для каждой категории
shapiro_low_steam = shapiro(low_danger_steam)
shapiro_medium_steam = shapiro(medium_danger_steam)
shapiro_high_steam = shapiro(high_danger_steam)

print('Гипотеза 5: Входное количество пара влияет на долю опасного газа')
# print(f"Тест Шапиро-Уилка для низкой опасности: {shapiro_low_steam}")
# print(f"Тест Шапиро-Уилка для средней опасности: {shapiro_medium_steam}")
# print(f"Тест Шапиро-Уилка для высокой опасности: {shapiro_high_steam}")

if shapiro_low_steam.pvalue > 0.05 and shapiro_medium_steam.pvalue > 0.05 and shapiro_high_steam.pvalue > 0.05:
    # Если данные нормальны, применяем t-тест
    t_stat_low_medium, p_value_low_medium = ttest_ind(low_danger_steam, medium_danger_steam)
    t_stat_medium_high, p_value_medium_high = ttest_ind(medium_danger_steam, high_danger_steam)

    print("t-статистика для низкой и средней опасности:", t_stat_low_medium)
    print("p-значение для низкой и средней опасности:", p_value_low_medium)
    print("t-статистика для средней и высокой опасности:", t_stat_medium_high)
    print("p-значение для средней и высокой опасности:", p_value_medium_high)

else:
    # Если данные не нормальны, используем Манн-Уитни
    mann_whitney_stat_low_medium, p_value_low_medium = mannwhitneyu(low_danger_steam, medium_danger_steam)
    mann_whitney_stat_medium_high, p_value_medium_high = mannwhitneyu(medium_danger_steam, high_danger_steam)

    print("Статистика Манна-Уитни для низкой и средней опасности:", mann_whitney_stat_low_medium)
    print("p-value Манна-Уитни для низкой и средней опасности:", p_value_low_medium)
    print("Статистика Манна-Уитни для средней и высокой опасности:", mann_whitney_stat_medium_high)
    print("p-value Манна-Уитни для средней и высокой опасности:", p_value_medium_high)

if p_value_low_medium < 0.05 or p_value_medium_high < 0.05:
    print("Мы отвергаем нулевую гипотезу:  входное количество пара влияет на долю опасного газа.")
else:
    print("Мы не отвергаем нулевую гипотезу: входное количество пара не влияет на долю опасного газа.")
print('\n')

# Гипотеза 6: Среднее содержание сухого остатка влияет на долю опасного газа
# Разделим данные по категориям опасности для среднего содержания сухого остатка
low_residue = data[data['danger_category'] == 'низкая']['stage_4_output_dry_residue_avg']
medium_residue = data[data['danger_category'] == 'средняя']['stage_4_output_dry_residue_avg']
high_residue = data[data['danger_category'] == 'высокая']['stage_4_output_dry_residue_avg']

# Проверим нормальность распределения для каждой категории
shapiro_low_dry = shapiro(low_residue)
shapiro_medium_dry = shapiro(medium_residue)
shapiro_high_dry = shapiro(high_residue)

print('Гипотеза 6: Среднее содержание сухого остатка влияет на долю опасного газа')
# print(f"Тест Шапиро-Уилка для низкой опасности: {shapiro_low_dry}")
# print(f"Тест Шапиро-Уилка для средней опасности: {shapiro_medium_dry}")
# print(f"Тест Шапиро-Уилка для высокой опасности: {shapiro_high_dry}")

if shapiro_low_dry.pvalue > 0.05 and shapiro_medium_dry.pvalue > 0.05 and shapiro_high_dry.pvalue > 0.05:
    t_stat_low_medium, p_value_low_medium = ttest_ind(low_residue, medium_residue)
    t_stat_medium_high, p_value_medium_high = ttest_ind(medium_residue, high_residue)

    print("t-статистика для низкой и средней опасности:", t_stat_low_medium)
    print("p-значение для низкой и средней опасности:", p_value_low_medium)
    print("t-статистика для средней и высокой опасности:", t_stat_medium_high)
    print("p-значение для средней и высокой опасности:", p_value_medium_high)
else:
    mann_whitney_stat_low_medium, p_value_low_medium = mannwhitneyu(low_residue, medium_residue)
    mann_whitney_stat_medium_high, p_value_medium_high = mannwhitneyu(medium_residue, high_residue)

    print("Статистика Манна-Уитни для низкой и средней опасности:", mann_whitney_stat_low_medium)
    print("p-value Манна-Уитни для низкой и средней опасности:", p_value_low_medium)
    print("Статистика Манна-Уитни для средней и высокой опасности:", mann_whitney_stat_medium_high)
    print("p-value Манна-Уитни для средней и высокой опасности:", p_value_medium_high)

# Выводы
if p_value_low_medium < 0.05 or p_value_medium_high < 0.05:
    print("Мы отвергаем нулевую гипотезу: среднее содержание сухого остатка влияет на долю опасного газа.")
else:
    print("Мы не отвергаем нулевую гипотезу: среднее содержание сухого остатка не влияет на долю опасного газа..")

# Гипотеза 7: Количество выходного количества продукта связано с категорией опасного газа
# Разделим данные по категориям опасности для полимера
low_polymer = data[data['danger_category'] == 'низкая']['stage_4_input_polymer']
medium_polymer = data[data['danger_category'] == 'средняя']['stage_4_input_polymer']
high_polymer = data[data['danger_category'] == 'высокая']['stage_4_input_polymer']

# Проверка нормальности
shapiro_low_polymer = shapiro(low_polymer)
shapiro_medium_polymer = shapiro(medium_polymer)
shapiro_high_polymer = shapiro(high_polymer)

print("Гипотеза 7: Количество выходного количества продукта связано с категорией опасного газа")
if shapiro_low_polymer.pvalue > 0.05 and shapiro_medium_polymer.pvalue > 0.05 and shapiro_high_polymer.pvalue > 0.05:
    # Если данные нормальны, применяем t-тест
    t_stat_low_medium, p_value_low_medium = ttest_ind(low_polymer, medium_polymer)
    t_stat_medium_high, p_value_medium_high = ttest_ind(medium_polymer, high_polymer)

    print("t-статистика для низкой и средней опасности:", t_stat_low_medium)
    print("p-значение для низкой и средней опасности:", p_value_low_medium)
    print("t-статистика для средней и высокой опасности:", t_stat_medium_high)
    print("p-значение для средней и высокой опасности:", p_value_medium_high)

else:
    # Если данные не нормальны, используем Манн-Уитни
    mann_whitney_stat_low_medium, p_value_low_medium = mannwhitneyu(low_polymer, medium_polymer)
    mann_whitney_stat_medium_high, p_value_medium_high = mannwhitneyu(medium_polymer, high_polymer)

    print("Статистика Манна-Уитни для низкой и средней опасности:", mann_whitney_stat_low_medium)
    print("p-value Манна-Уитни для низкой и средней опасности:", p_value_low_medium)
    print("Статистика Манна-Уитни для средней и высокой опасности:", mann_whitney_stat_medium_high)
    print("p-value Манна-Уитни для средней и высокой опасности:", p_value_medium_high)

if p_value_low_medium < 0.05 or p_value_medium_high < 0.05:
    print("Мы отвергаем нулевую гипотезу: количество выходного продукта между категориями опасности различается.")
else:
    print("Мы не отвергаем нулевую гипотезу: количество выходного продукта между категориями опасности не различается.")
print('\n')

# Дополнительная проверка всех параметров на влияние
numeric_columns = data.select_dtypes(include=[np.number]).columns.to_list()
diff = []
for i in numeric_columns:
  # Разделим данные по категориям опасности для температуры
  low = data[data['danger_category'] == 'низкая'][i]
  medium = data[data['danger_category'] == 'средняя'][i]
  high = data[data['danger_category'] == 'высокая'][i]

  # Проверка нормальности
  shapiro_low = shapiro(low)
  shapiro_medium = shapiro(medium)
  shapiro_high = shapiro(high)

  if shapiro_low.pvalue > 0.05 and shapiro_medium.pvalue > 0.05 and shapiro_high.pvalue > 0.05:
      # Если данные нормальны, применяем t-тест
      t_stat_low_medium, p_value_low_medium = ttest_ind(low, medium)
      t_stat_medium_high, p_value_medium_high = ttest_ind(medium, high)

      # print("t-статистика для низкой и средней опасности по температуре:", t_stat_low_medium)
      # print("p-значение для низкой и средней опасности по температуре:", p_value_low_medium)
      # print("t-статистика для средней и высокой опасности по температуре:", t_stat_medium_high)
      # print("p-значение для средней и высокой опасности по температуре:", p_value_medium_high)

  else:
      # Если данные не нормальны, используем Манн-Уитни
      mann_whitney_stat_low_medium, p_value_low_medium = mannwhitneyu(low, medium)
      mann_whitney_stat_medium_high, p_value_medium_high = mannwhitneyu(medium, high)

      # print("Статистика Манна-Уитни для низкой и средней опасности по температуре:", mann_whitney_stat_low_medium)
      # print("p-value Манна-Уитни для низкой и средней опасности по температуре:", p_value_low_medium)
      # print("Статистика Манна-Уитни для средней и высокой опасности по температуре:", mann_whitney_stat_medium_high)
      # print("p-value Манна-Уитни для средней и высокой опасности по температуре:", p_value_medium_high)

  if p_value_low_medium < 0.05 or p_value_medium_high < 0.05:
      # print(f"Мы отвергаем нулевую гипотезу: {i} между категориями опасности различается.")
      diff.append(i)
  else:
      # print(f"Мы не отвергаем нулевую гипотезу: {i} между категориями опасности не различается.")
      continue
  # print('\n')
print("Параметры имеющие влияние на долю опасного газа:")
print(*diff, sep='\n')

data['DateTime'] = pd.to_datetime(data['DateTime'])

# Рассмотрим данные от 12.2021
filtered_df = data[(data['DateTime'].dt.month == 12) & (data['DateTime'].dt.year == 2021)]


numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.to_list()
diff = []
for i in numeric_columns:
  # Разделим данные по категориям опасности для температуры
  low = filtered_df[filtered_df['danger_category'] == 'низкая'][i]
  medium = filtered_df[filtered_df['danger_category'] == 'средняя'][i]
  high = filtered_df[filtered_df['danger_category'] == 'высокая'][i]

  # Проверка нормальности
  # shapiro_low = shapiro(low)
  shapiro_medium = shapiro(medium)
  shapiro_high = shapiro(high)

  if shapiro_medium.pvalue > 0.05 and shapiro_high.pvalue > 0.05:
      # Если данные нормальны, применяем t-тест
      t_stat_medium_high, p_value_medium_high = ttest_ind(medium, high)


  else:
      # Если данные не нормальны, используем Манн-Уитни
      mann_whitney_stat_medium_high, p_value_medium_high = mannwhitneyu(medium, high)

  if p_value_medium_high < 0.05:
      # print(f"Мы отвергаем нулевую гипотезу: {i} между категориями опасности различается.")
      diff.append(i)
  else:
      # print(f"Мы не отвергаем нулевую гипотезу: {i} между категориями опасности не различается.")
      continue
print("Параметры имеющие влияние на долю опасного газа 12.2021:")
print(*diff, sep='\n')
