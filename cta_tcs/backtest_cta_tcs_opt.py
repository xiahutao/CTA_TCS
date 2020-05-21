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
            # analysis_obj.plot_cumsum_pnl(show=False,
            #                              title='_'.join([strategy_obj._strategy_name, '_'.join(strategy_obj._symbols)]))
            # analysis_obj.plot_all()
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

    pool = Pool(max(1, cpu_count() - 11))
    pool = Pool(5)

    file_save_obj = file_saver()
    # symbols = ['AP', 'AL', 'SC', 'HC', 'MA', 'J', 'TA', 'PP', 'I', 'ZC', ]  # 不加止损
    # symbols = ['SC', 'I', 'TA', 'MA', 'ZC', 'J', 'PP', 'AL', 'IF', 'RB', 'CF']  # 加止损（25， 2.2）

    class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal', 'SoftComm', 'Food']
    # class_lst = ['SoftComm']
    symbols_dict = {'Grains': ['C', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM'],
                    'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU'],
                    'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],
                    'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM', 'SP'],
                    'Equity': ['IF', 'IH', 'IC'],
                    'Bonds': ['T', 'TF'],
                    'PreciousMetal': ['AG', 'AU'],
                    'SoftComm': ['CF', 'CS', 'SR'],
                    'Food': ['JD', 'AP', 'CJ'],
                    'all': ['A', 'AG', 'AL', 'AP', 'AU', 'B', 'BU', 'C', 'CF', 'CS', 'CU', 'CY', 'FG',
                            'HC', 'I', 'IC', 'IF', 'IH', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P',
                            'PB', 'PP', 'RB', 'RM', 'RU', 'SC', 'SF', 'SM', 'SN', 'SR',
                            'T', 'TA', 'TF', 'V', 'Y', 'ZC', 'ZN']}  # 所有品种
    ma_period_lst = [i for i in range(5, 105, 5)]
    k_period_lst = [i/10 for i in range(5, 51, 1)]
    for cls in class_lst:
        symbols = symbols_dict[cls]
        symbols = [i + '_VOL' for i in symbols]
        run_symbols_0 = [symbols]
        for ma_period in ma_period_lst:
            for k in k_period_lst:
                for each in run_symbols_0:
                    run_symbols = each
                    run_params = {'capital': 400000000,
                                  'daily_start_time': '9:00:00',
                                  'daily_end_time': '23:30:00',
                                  'start_date': '20090101',
                                  'end_date': '20200623'
                                  }

                    strategy_params = {'period': 1440,
                                       'ma_period': [ma_period],
                                       'k': [k],
                                       'maxLeverage': 4,
                                       'targetVol': 0.1,
                                       'volLookback': 20,
                                       }
                    exec_lag = 1
                    # result_folder = 'e://Strategy//TCS//resRepo_all_%s_%s_exec_%s_%s' % (
                    #     strategy_params['ma_period'], int(10 * strategy_params['k']), '_'.join(i[:-4] for i in run_symbols),
                    #     exec_lag)
                    result_folder = 'e://Strategy//TCS//better//resRepo_ymjh_%s_%s_%s' % (cls, ma_period, int(k*10))
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
