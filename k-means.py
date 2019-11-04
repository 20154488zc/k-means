from math import pow, sqrt
from random import sample
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
import os


def dtw_distance_new(ts_a, ts_b, d=lambda x, y: abs(x - y)):
    M, N = len(ts_a), len(ts_b)
    cost_sum = []
    for dim in range(4):
        cost = np.zeros((M, N))
        cost[0, 0] = d(ts_a[0][dim], ts_b[0][dim])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i][dim], ts_b[0][dim])

        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0][dim], ts_b[j][dim])

        for i in range(1, M):
            for j in range(1, N):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i][dim], ts_b[j][dim])
        cost_sum.append(cost[-1, -1])
    cost_average = (cost_sum[0] + cost_sum[1] + cost_sum[2] + cost_sum[3])/4
    # cost_average = (cost_sum[0] + cost_sum[1])
    # cost_average = (cost_sum[2] + cost_sum[3])/2
    return cost_average


def dtw_distance_1dim(ts_a, ts_b, d=lambda x, y: abs(x - y)):
    M, N = len(ts_a), len(ts_b)
    cost = np.zeros((M, N))

    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    for i in range(1, M):
        for j in range(1, N):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
    return cost[-1, -1]


def average_points(list_of_series):
    print(np.array(list_of_series).shape)
    aver = np.zeros((len(list_of_series[0]), 4))
    for dim in range(4):
        for j in range(len(list_of_series[0])):
            total = 0
            for i in range(len(list_of_series)):
                total += list_of_series[i][j][dim]
            aver[j][dim] = (total/len(list_of_series))
    return aver


def average_points_1dim(list_of_series):
    print(np.array(list_of_series).shape)
    if(list_of_series):
        aver = np.zeros((len(list_of_series[0])))
        for j in range(len(list_of_series[0])):
            total = 0
            for i in range(len(list_of_series)):
                total +=list_of_series[i][j]
            aver[j] = (total/len(list_of_series))
    else:
        aver = np.zeros((37))
    return aver


# function returns a list of points which are the centers
def k_means(points, K):
    above_distance = True
    count = 0
    m = len(points)
    label_pred = np.zeros((m))
    new_means = sample(list(points), K)
    mean_points = {}
    while above_distance:
        for i in range(K):
            mean_points[i] = []
        # above_distance = True
        for point in range(m):
            min_dist = float('inf')
            min_mean = new_means[0]
            for mean in range(K):
                dist = dtw_distance_new(points[point], new_means[mean])
                if dist < min_dist:
                    min_dist = dist
                    min_mean = mean
            label_pred[point] = min_mean
            mean_points[min_mean].append(points[point])
        old_means = new_means.copy()
        new_means.clear()
        for key in range(K):
            new_means.append(average_points(mean_points[key]))
        if((np.array(new_means) == np.array(old_means)).all() or count == 100):
            above_distance = False
        print(count)
        count += 1
    means_figure(np.array(new_means), K)
    new_figure(label_pred, K)
    means_dtw(np.array(new_means), label_pred, K)
    return new_means, label_pred


data = np.load('dataset_all.npy')
label = np.load('dataset_label_all.npy')
data0 = np.zeros((len(data), len(data[0])))
for i in range(len(data)):
    for j in range(len(data[i])):
        data0[i][j] = data[i][j][0]
data0 = preprocessing.MinMaxScaler().fit_transform(data0)
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j][0] = data0[i][j]

data1 = np.zeros((len(data), len(data[0])))
for i in range(len(data)):
    for j in range(len(data[i])):
        data1[i][j] = data[i][j][1]
data1 = preprocessing.MinMaxScaler().fit_transform(data1)
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j][1] = data1[i][j]

data2 = np.zeros((len(data), len(data[0])))
for i in range(len(data)):
    for j in range(len(data[i])):
        data2[i][j] = data[i][j][2]
data2 = preprocessing.MinMaxScaler().fit_transform(data2)
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j][2] = data2[i][j]

data3 = np.zeros((len(data), len(data[0])))
for i in range(len(data)):
    for j in range(len(data[i])):
        data3[i][j] = data[i][j][3]
data3 = preprocessing.MinMaxScaler().fit_transform(data3)
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j][3] = data3[i][j]

def figure():
    x = np.arange(0,37)
    for i in range(len(data)):
        plt.figure(figsize=(10, 5), dpi=120)
        plt.plot(x, data0[i], color='blue',  linewidth=1.5)
        plt.plot(x, data2[i], color='red', linewidth=1.5)
        plt.plot(x, data1[i], color='green',  linewidth=1.5)
        plt.plot(x, data3[i], color='black', linewidth=1.5)
        plt.xlim(0, 38)
        plt.ylim(0, 1)
        title = 'NO.'+str(i) + '     label:' + str(label[i])
        plt.title(title)
        plt.grid()
        path = 'figure/' + str(i) + '.png'
        plt.savefig(path)
        plt.close()


def dtw_figure():
    x = np.arange(0, len(data))
    for num in range(len(data)):
        dtw = []
        for i in range(len(data)):
            dtw.append(dtw_distance_new(data[i], data[num]))
        plt.figure(figsize=(20, 5))
        plt.bar(x, dtw, align='center')
        # plt.hist(dtw, bins=len(data), normed=0, facecolor='green', edgecolor='black', alpha=0.6)
        title = 'DTW of NO.'+str(num) + '   and others'
        plt.title(title)
        path = 'dtw_figure/' + str(num) + '.png'
        plt.savefig(path)
        plt.close()


def new_figure(label_pred, K):
    x = np.arange(0, 37)
    for i in range(len(data)):
        plt.figure(figsize=(10, 5), dpi=120)
        plt.plot(x, data0[i], color='blue',  linewidth=1.5)
        plt.plot(x, data1[i], color='green',  linewidth=1.5)
        plt.plot(x, data2[i], color='red', linewidth=1.5)
        plt.plot(x, data3[i], color='black', linewidth=1.5)
        plt.xlim(0, 38)
        plt.ylim(0, 1)
        title = 'NO.'+str(i) + '     label:' + str(label[i])
        plt.title(title)
        plt.grid()
        path = 'new_figure/' + str(K) + '_means/' + str(int(label_pred[i]))
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + '/' + str(i) + '.png'
        plt.savefig(path)
        plt.close()


def means_figure(means, K):
    means0 = np.zeros((len(means), len(means[0])))
    for i in range(len(means)):
        for j in range(len(means[i])):
            means0[i][j] = means[i][j][0]
    means1 = np.zeros((len(means), len(means[0])))
    for i in range(len(means)):
        for j in range(len(means[i])):
            means1[i][j] = means[i][j][1]
    means2 = np.zeros((len(means), len(means[0])))
    for i in range(len(means)):
        for j in range(len(means[i])):
            means2[i][j] = means[i][j][2]
    means3 = np.zeros((len(means), len(means[0])))
    for i in range(len(means)):
        for j in range(len(means[i])):
            means3[i][j] = means[i][j][3]
    x = np.arange(0, 37)
    path = 'means/' + str(K)
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(means)):
        plt.figure(figsize=(10, 5), dpi=120)
        plt.plot(x, means0[i], color='blue', linewidth=1.5)
        plt.plot(x, means1[i], color='green', linewidth=1.5)
        plt.plot(x, means2[i], color='red', linewidth=1.5)
        plt.plot(x, means3[i], color='black', linewidth=1.5)
        plt.xlim(0, 38)
        plt.ylim(0, 1)
        plt.grid()
        path = 'means/' + str(K) + '/' + str(i) + '.png'
        plt.savefig(path)
        plt.close()


def means_dtw(means, label_pred, K):
    x = np.arange(0, len(data))
    for mean in range(len(means)):
        dtw = []
        for num in range(len(data)):
            if label_pred[num] == mean:
                dtw.append(dtw_distance_new(means[mean], data[num]))
        first_num = len(dtw)
        for num in range(len(data)):
            if label_pred[num] != mean:
                dtw.append(dtw_distance_new(means[mean], data[num]))
        plt.figure(figsize=(20, 5))
        plt.bar(x, dtw, align='center')
        title = 'first ' + str(first_num) + ' are in mean ' + str(mean)
        plt.title(title)
        path = 'means/' + str(K) + '/means_' + str(mean) + '.png'
        plt.savefig(path)
        plt.close()


# data = np.load('dataset_all.npy')
K = 4
for K in range(8,10):
    means, label_pred = k_means(data, K)
# new_figure(label_pred)
# plt.figure(figsize=(20, 5))
# x = np.arange(0, len(label_pred))
# plt.bar(x, label_pred, align='center')
# plt.show()


# figure()
# data = data2.copy()
# dtw_figure()
# dtw_sum = 0
# for num in range(len(data)):
#     for i in range(len(data)):
#         dtw_sum += dtw_distance_1dim(data3[i], data3[num])
# dtw_average = dtw_sum/((len(data)-1)*(len(data)))