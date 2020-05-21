# -*- coding: utf-8 -*-
# @Time    : 2020/2/18 16:50
# 每两年调一次参数，每个月回看两年加几个月
# @Author  : zhangfang
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
import json
import seaborn as sns
# import tensorflow as tf
# tf.__version__
def makeHeatMap(path, chatPath, pair, dirs, aft):
    """"""
    pnl_list = list()
    for d in dirs:
        if '_' + pair in d:
            speed = json.loads(d.split('_speeds_')[1])[0]
            pnl_df = pd.read_csv(path + '/' + d + f'/daily_pnl{aft}.csv', index_col=0)
            pnl_df.index = pd.to_datetime(pnl_df.index).tz_convert('PRC')
            pnl_df['year'] = pnl_df.index.year
            pnl_year = pnl_df.groupby(by=['year'])[f'daily_pnl{aft}'].sum()
            pnl_year.name = speed
            pnl_year = pd.DataFrame(pnl_year)
            pnl_list.append(pnl_year)
            del pnl_df
    pnl_data = pd.concat(pnl_list, axis=1)
    pnl_data.sort_index(axis=1, ascending=False, inplace=True)
    pnl_data = pnl_data / 10000000
    del pnl_list
    f, ax1 = plt.subplots(figsize=(len(pnl_data), len(pnl_data.columns)))
    # pnl_data = pnl_data.corr()  # pt为数据框或者是协方差矩阵
    vmax = np.abs(pnl_data).max().max()
    vmin = -vmax
    sns.heatmap(pnl_data.T, annot=True, ax=ax1, vmax=vmax, vmin=vmin, annot_kws={'weight': 'bold', 'color': 'blue'},
                cmap='rainbow')
    ax1.set_title(pair + aft)
    ax1.set_ylabel('speed')
    plt.savefig(chatPath + f'/{pair}{aft}_ThermodynamicChart.png')
    print(f'/{pair}{aft}_ThermodynamicChart.png', '完成')


def yearsharpRatio(netlist, n):
    '''
    :param netlist:
    :param n: 每交易日对应周期数
    :return:
    '''
    row = []
    new_lst = copy.deepcopy(netlist)
    new_lst = [new_lst[i] for i in range(0, len(new_lst), n)]
    for i in range(1, len(new_lst)):
        row.append(math.log(new_lst[i] / new_lst[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(252, 0.5)


if __name__ == "__main__":
    class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal', 'SoftComm', 'Food']
    symbols_dict = {'Grains': ['C', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM'],
                    'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU'],
                    'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],
                    'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM'],
                    'Equity': ['IF', 'IH', 'IC'],
                    'Bonds': ['T', 'TF'],
                    'PreciousMetal': ['AG', 'AU'],
                    'SoftComm': ['CF', 'CS', 'SR'],
                    'Food': ['JD', 'AP', 'CJ'],
                    }
    # class_lst = ['SoftComm']
    # time_list = [('2010-01-01', '2011-01-01'), ('2011-01-01', '2012-01-01'), ('2012-01-01', '2013-01-01'),
    #              ('2013-01-01', '2014-01-01'), ('2014-01-01', '2015-01-01'), ('2015-01-01', '2016-01-01'),
    #              ('2016-01-01', '2017-01-01'), ('2017-01-01', '2018-01-01'), ('2018-01-01', '2019-01-01'),
    #              ('2019-01-01', '2020-01-01')]
    time_list = [('2010-01-01', '2012-01-01'), ('2012-01-01', '2014-01-01'), ('2014-01-01', '2016-01-01'),
                 ('2016-01-01', '2018-01-01'), ('2018-01-01', '2020-01-01')]
    chatPath = 'e://Strategy//tcs//fig//'
    state_name = 'sharp'

    ma_period_lst = [i for i in range(100, 4, -5)]
    k_period_lst = [i / 10 for i in range(5, 51, 2)]

    # s_period_lst = [i for i in range(29, 3, -1)]
    # l_period_lst = [i for i in range(14, 68, 3)]

    lst = []
    for clas in class_lst:
        for (s_date, e_date) in time_list:
            best_sharp = 0
            state = []
            harvest = []
            s_period_best = 30
            l_period_best = 1.5

            for i in range(1, len(ma_period_lst)-1):
                s_period = ma_period_lst[i]
                for j in range(1, len(k_period_lst)-1):
                    l_period = k_period_lst[j]
                    sharp_lst = []
                    for m in range(i-1, i+2):
                        temp_s_period = ma_period_lst[m]
                        for n in range(j-1, j+2):
                            try:
                                temp_l_period = k_period_lst[n]
                                result_folder = 'e://Strategy//TCS//better//resRepo_ymjh_%s_%s_%s' % (
                                clas, temp_s_period, int(10 * temp_l_period))
                                daily_returns = pd.read_csv(result_folder + '//daily_returns.csv', header=None)
                                daily_returns.columns = ['trade_date', 'daily_return']
                                temp = daily_returns[
                                    (daily_returns['trade_date'] >= s_date) & (daily_returns['trade_date'] < e_date)]
                                try:
                                    sharp = np.mean(temp.daily_return) / np.std(temp.daily_return) * math.pow(252, 0.5)
                                except Exception as e:
                                    print(str(e))
                                    sharp = -1
                                print(sharp)
                                sharp_lst.append(sharp)
                            except Exception as e:
                                print(str(e))
                                continue
                    if np.mean(sharp_lst) > best_sharp:
                        best_sharp = np.mean(sharp_lst)
                        # print(sharp_lst)
                        # print(best_sharp)
                        s_period_best = s_period
                        l_period_best = l_period
            row = []
            row.append(clas)
            row.append(s_date)
            row.append(e_date)
            row.append(s_period_best)
            row.append(l_period_best)
            lst.append(row)
    ret = pd.DataFrame(lst, columns=['class', 's_date', 'e_date', 'ma_period', 'k'])
    print(ret)
    ret.to_csv('e:/Strategy/TCS/para/' + 'para_opt_history9.csv')



