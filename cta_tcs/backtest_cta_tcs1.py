# -*- coding: utf-8 -*-
import sys
import os

CurrentPath = os.path.dirname(__file__)
print(CurrentPath)
sys.path.append(CurrentPath.replace('cta_momentum', ''))
print(sys.path)
import datetime
import warnings
import time

warnings.filterwarnings("ignore")
import traceback
import pandas
from execution.execution import Execution
from analysis.analysis import Analysis
from cta_tcs.ctatcs import CtaTcsStrategy
from settlement.settlement import Settlement
from data_engine.data_factory import DataFactory
import data_engine.setting as setting

from common.file_saver import file_saver
from common.os_func import check_fold
from data_engine.setting import ASSETTYPE_FUTURE, FREQ_1M, FREQ_5M, FREQ_1D


# DataFactory.sync_future_from_remote()


def run_backtest(run_symbols, freq, result_folder, strategy_params, run_params, exec_lag, saving_file=False):
    import time
    t1 = time.clock()
    try:
        DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
        # 策略对象
        strategy_obj = CtaTcsStrategy(symbols_list=run_symbols,
                                      freq=freq,
                                      asset_type=ASSETTYPE_FUTURE,
                                      result_fold=result_folder,
                                      **strategy_params
                                      )
        # 回测
        signal_dataframe = strategy_obj.run_test(startDate=run_params['start_date'], endDate=run_params['end_date'],
                                                 **run_params
                                                 )

        execution_obj = Execution(freq=freq, exec_price_mode=Execution.EXEC_BY_OPEN, exec_lag=exec_lag)
        (success, positions_dataframe) = execution_obj.exec_trading(signal_dataframe=signal_dataframe)

        if not success:
            print(positions_dataframe)
            assert False

        if success:
            settlement_obj = Settlement(init_aum=run_params['capital'])
            # file_saver().save_file(positions_dataframe, os.path.join(result_folder, 'positions_dataframe.csv'))
            settlement_obj.settle(positions_dataframe=positions_dataframe)
            print(settlement_obj.daily_return)

            # 分析引擎，  结果保存到result_folder文件夹下
            analysis_obj = Analysis(daily_returns=settlement_obj.daily_return_by_init_aum,
                                    daily_positions=settlement_obj.daily_positions,
                                    daily_pnl=settlement_obj.daily_pnl,
                                    daily_pnl_gross=settlement_obj.daily_pnl_gross,
                                    daily_pnl_fee=settlement_obj.daily_pnl_fee,
                                    transactions=settlement_obj.transactions,
                                    round_trips=settlement_obj.round_trips,
                                    result_folder=result_folder,
                                    strategy_id='_'.join(
                                        [strategy_obj._strategy_name, '_'.join(strategy_obj._symbols)]),
                                    symbols=strategy_obj._symbols,
                                    strategy_type=strategy_obj._strategy_name)
            sharpe_ratio = analysis_obj.sharpe_ratio()
            sharpe_dataframe = pandas.DataFrame({'symbol': ['_'.join(run_symbols)], 'sharp': [sharpe_ratio]})
            file_saver().save_file(sharpe_dataframe, os.path.join(result_folder, 'sharpe_dataframe.csv'))
            analysis_obj.plot_cumsum_pnl(show=False,
                                         title='_'.join([strategy_obj._strategy_name, '_'.join(strategy_obj._symbols)]))
            analysis_obj.plot_all()
            analysis_obj.save_result()
    except:
        if saving_file:
            file_saver().join()
        traceback.print_exc()
    if saving_file:
        file_saver().join()
    print('=================', 'run_backtest', '%.6fs' % (time.clock() - t1))


def PowerSetsRecursive(items):
    # 求集合的所有子集
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result


if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_LOCAL)
    client = DataFactory.get_mongo_client()
    print(client.database_names())
    from multiprocessing import Pool, cpu_count

    pool = Pool(max(1, cpu_count() - 2))
    method = ''
    file_save_obj = file_saver()
    # symbols = ['AP', 'AL', 'SC', 'HC', 'MA', 'J', 'TA', 'PP', 'I', 'ZC', ]  # 不加止损
    # symbols = ['SC', 'I', 'TA', 'MA', 'ZC', 'J', 'PP', 'AL', 'IF', 'RB', 'CF']  # 加止损（25， 2.2）
    symbols_all = ['C', 'CS', 'A', 'B', 'M', 'RM', 'Y', 'P', 'OI', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA',
                   'SC', 'FU', 'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG',
                   'SF', 'SM', 'SP', 'IF', 'IH', 'IC', 'T', 'TF', 'AG', 'AU', 'CF', 'SR', 'JD',
                   'AP', 'CJ']
    symbols_all = ['SC', 'CJ', 'SM', 'AG', 'TA', 'TF', 'P', 'NI', 'RU', 'IC', 'B', 'AP', 'BU', 'I', 'C', 'MA', 'PB',
                   'JM', 'SF', 'SR', 'V', 'FU', 'IH', 'AL', 'ZC', 'FG', 'RM', 'JD', 'PP']
    symbols_all = ['SC', 'J', 'TA', 'I', 'MA', 'RU', 'ZC', 'P', 'SF', 'CF', 'NI', 'TF', 'IF', 'SM', 'PB', 'T', 'SR',
                   'AL', 'BU', 'AU', 'AP', 'FU', 'C']

    class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal', 'all']
    class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal', 'LightIndustry',
                 'Oil', 'Grease', 'SoftComm', 'Food']
    symbols_dict = {'Grains': ['C', 'A', 'B'],  # 农产品
                    'Grease': ['M', 'RM', 'Y', 'P', 'OI'],  # 油脂油料
                    'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA'],  # 化工
                    'Oil': ['SC', 'FU'],
                    'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],  # 金属
                    'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'SF', 'SM'],  # 黑色系
                    'LightIndustry': ['SP', 'FG'],  # 轻工业
                    'Equity': ['IF', 'IH', 'IC'],  # 股指
                    'Bonds': ['T', 'TF'],  # 债券
                    'PreciousMetal': ['AG', 'AU'],  # 贵金属
                    'SoftComm': ['CF', 'CS', 'SR'],  # 软商品
                    'Food': ['JD', 'AP', 'CJ'],  # 农副产品
                    'all': ['C', 'CS', 'A', 'B', 'M', 'RM', 'Y', 'P', 'OI', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA',
                            'SC', 'FU', 'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG',
                            'SF', 'SM', 'SP', 'FG', 'IF', 'IH', 'IC', 'T', 'TF', 'AG', 'AU', 'CF', 'CS', 'SR', 'JD',
                            'AP', 'CJ']}  # 所有品种
    ma_period_dict = {'Grains': [10, 10, 15, 15, 20],
                      'Chem': [30, 70, 45, 25, 15],
                      'BaseMetal': [10, 15, 35, 15, 30],
                      'Bulks': [35, 20, 95, 15, 35],
                      'Equity': [10, 15, 45, 95, 80],
                      'Bonds': [20, 35, 80, 30],
                      'PreciousMetal': [25, 20, 10, 10, 20],
                      'LightIndustry': [10, 25, 95, 55],
                      'Oil': [10, 75, 20, 40, 10],
                      'Grease': [10, 95, 10, 15, 65],
                      'SoftComm': [15, 55, 20, 25, 95],
                      'Food': [10, 85, 70, 30]}
    k_dict = {'Grains': [0.7, 0.7, 2.1, 4.7, 1.1],
              'Chem': [3.9, 0.7, 1.7, 3.5, 0.7],
              'BaseMetal': [1.3, 2.1, 2.5, 4.7, 2.3],
              'Bulks': [1.3, 1.5, 4.7, 2.3, 3.9],
              'Equity': [1.7, 1.7, 2.5, 4.7, 4.7],
              'Bonds': [3.7, 4.1, 4.7, 3.7],
              'PreciousMetal': [0.9, 0.9, 0.7, 0.9, 2.5],
              'LightIndustry': [1.9, 4.7, 4.7, 1.9],
              'Oil': [0.7, 4.7, 2.7, 1.7, 0.7],
              'Grease': [0.7, 4.3, 2.3, 4.5, 3.3],
              'SoftComm': [1.5, 2.1, 4.5, 1.3, 4.7],
              'Food': [1.1, 0.7, 3.1, 0.9]}
    ma_period_dic = {}
    k_period_dic = {}
    for clas in class_lst:
        symbols = symbols_dict[clas]
        for symbol in symbols:
            ma_period_dic[symbol] = ma_period_dict[clas]
            k_period_dic[symbol] = k_dict[clas]
    symbols = [i + '_VOL' for i in symbols_all]
    run_symbols_0 = [symbols]
    for each in run_symbols_0:
        run_symbols = each
        run_params = {'capital': 400000000,
                      'daily_start_time': '9:00:00',
                      'daily_end_time': '23:30:00',
                      'start_date': '20170101',
                      'end_date': '20200430'
                      }

        strategy_params = {'period': 1440,
                           'ma_period': ma_period_dic,
                           'k': k_period_dic,
                           'targetVol': 0.1,
                           'volLookback': 20,
                           'maxLeverage': 4,
                           }
        if method == 'ori':
            strategy_params = {'period': 1440,
                               'ma_period': [30],
                               'k': [1.5],
                               'targetVol': 0.1,
                               'volLookback': 20,
                               'maxLeverage': 4,
                               }
        exec_lag = 1
        # result_folder = 'e://Strategy//TCS//resRepo_all_%s_%s_exec_%s_%s' % (
        #     strategy_params['ma_period'], int(10 * strategy_params['k']), '_'.join(i[:-4] for i in run_symbols),
        #     exec_lag)
        result_folder = 'e://Strategy//TCS//backtest//2017_mutipara_%s' % ('_'.join([i[:-4] for i in run_symbols]))
        check_fold(result_folder)
        freq = FREQ_1M
        if strategy_params['period'] == 5:
            freq = FREQ_5M
        elif strategy_params['period'] == 1:
            freq = FREQ_1M
        else:
            freq = FREQ_1D
        try:
            # 策略对象
            print('symbols', '_'.join(each))
            pool.apply_async(run_backtest,
                             args=(run_symbols, freq, result_folder, strategy_params, run_params, exec_lag, True))
            # run_backtest(run_pairs, freq, result_folder, strategy_params, run_params, exec_lag,saving_file=False)
            DataFactory().clear_data()
        except:
            DataFactory().clear_data()
            traceback.print_exc()
    pool.close()
    file_saver().join()
    pool.join()
