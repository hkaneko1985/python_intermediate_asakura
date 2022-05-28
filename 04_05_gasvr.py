# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import random
import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools
from sklearn import model_selection
from sklearn import svm

# 設定 ここから
number_of_population = 100  # GA の個体数
number_of_generation = 150  # GA の世代数
fold_number = 5  # クロスバリデーションの fold 数
svr_c_2_range = (-5, 10)  # SVR の C の範囲 (2 の何乗か)
svr_epsilon_2_range = (-10, 0)  # SVR の epsilon の範囲 (2 の何乗か)
svr_gamma_2_range = (-20, 10)  # SVR の gamma の範囲 (2 の何乗か)
threshold_of_variable_selection = 0.5  # 染色体の 0, 1 を分ける閾値
# 設定 ここまで

probability_of_crossover = 0.5
probability_of_mutation = 0.2

dataset = pd.read_csv('selected_descriptors_with_boiling_point.csv', index_col=0)  # データセットの読み込み

y_train = dataset.iloc[:, 0]  # 目的変数
x_train = dataset.iloc[:, 1:]  # 説明変数

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)

# GASVR
creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # for minimization, set weights as (-1.0,)
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
min_boundary = np.zeros(x_train.shape[1] + 3)
max_boundary = np.ones(x_train.shape[1] + 3) * 1.0
min_boundary[-3] = svr_c_2_range[0]
min_boundary[-2] = svr_epsilon_2_range[0]
min_boundary[-1] = svr_gamma_2_range[0]
max_boundary[-3] = svr_c_2_range[1]
max_boundary[-2] = svr_epsilon_2_range[1]
max_boundary[-1] = svr_gamma_2_range[1]


def create_ind_uniform(min_boundary, max_boundary):
    index = []
    for min, max in zip(min_boundary, max_boundary):
        index.append(random.uniform(min, max))
    return index


toolbox.register('create_ind', create_ind_uniform, min_boundary, max_boundary)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.create_ind)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    individual_array = np.array(individual)
    selected_x_variable_numbers = np.where(individual_array[:-3] > threshold_of_variable_selection)[0]
    selected_autoscaled_x_train = autoscaled_x_train.iloc[:, selected_x_variable_numbers]
    if len(selected_x_variable_numbers):
        # cross-validation
        model_in_cv = svm.SVR(kernel='rbf', C=2 ** round(individual_array[-3]),
                              epsilon=2 ** round(individual_array[-2]), gamma=2 ** round(individual_array[-1]))
        estimated_y_train_in_cv = model_selection.cross_val_predict(model_in_cv, selected_autoscaled_x_train,
                                                                    autoscaled_y_train, cv=fold_number)
        estimated_y_train_in_cv = estimated_y_train_in_cv * y_train.std(ddof=1) + y_train.mean()
        value = 1 - sum((y_train - estimated_y_train_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)
    else:
        value = -999

    return value,


toolbox.register('evaluate', evalOneMax)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=3)

# random.seed(100)
random.seed()
pop = toolbox.population(n=number_of_population)

print('Start of evolution')

fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

print('  Evaluated %i individuals' % len(pop))

for generation in range(number_of_generation):
    print('-- Generation {0} --'.format(generation + 1))

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < probability_of_crossover:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < probability_of_mutation:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print('  Evaluated %i individuals' % len(invalid_ind))

    pop[:] = offspring
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print('  Min %s' % min(fits))
    print('  Max %s' % max(fits))
    print('  Avg %s' % mean)
    print('  Std %s' % std)

print('-- End of (successful) evolution --')

best_individual = tools.selBest(pop, 1)[0]
best_individual_array = np.array(best_individual)
selected_x_variable_numbers = np.where(best_individual_array[:-3] > threshold_of_variable_selection)[0]
selected_descriptors = x_train.iloc[:, selected_x_variable_numbers]
selected_descriptors.to_csv('gasvr_selected_x.csv')  # 保存
selected_hyperparameters = pd.DataFrame(np.round(best_individual_array[-3:]), index=['C', 'epsilon', 'gamma'], columns=['hyperparameters of SVR (log2)'])
selected_hyperparameters.to_csv('gasvr_selected_hyperparameters.csv')  # 保存
