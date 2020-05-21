# -*- coding: utf-8 -*-
"""
Created on Fri Dec  13 16:34:53 2019

@author: zhangfang
"""

import pandas as pd
import numpy  as np
import datetime
from common.file_saver import file_saver
from common.decorator import runing_time
from strategy.strategy import Strategy
from data_engine.global_variable import ASSETTYPE_FUTURE,DATASOURCE_REMOTE,DATASOURCE_LOCAL
from data_engine.data_factory import DataFactory
import talib
import math


class CtaTcsStrategy(Strategy):

    def __init__(self, asset_type, symbols_list, freq, ma_period, k, targetVol, volLookback,
                 period, result_fold, **kwargs):

        super(CtaTcsStrategy, self).__init__(period=period,
                                             ma_period=ma_period,
                                             k=k,
                                             targetVol=targetVol,
                                             volLookback=volLookback
                                             )
        self._strategy_name = 'CtaTcsStrategy'
        self._strategy_type = 'intraday_symbols'
        self._freq = freq
        self._asset_type = asset_type
        self._symbol_pair_list = symbols_list
        self.maxLeverage = 1

        self._symbols = set()
        for s1 in symbols_list:
            if s1 not in self._symbols:
                self._symbols.add(s1)

        self._contract_size_dict = {}
        self._tick_size_dict = {}

        self._market_data = None
        self.result_fold = result_fold

    def cap_vol_by_rolling(self, vol, targetvol):
        idxs = vol.index
        for idx in range(len(idxs)):
            curDate = idxs[idx]
            vol[curDate] = max(vol[curDate], targetvol)
        return vol

    def _get_market_info(self):
        print('================', self._symbols)
        self._contract_size_dict = DataFactory.get_contract_size_dict(symbols=list(self._symbols),
                                                                      asset_type=ASSETTYPE_FUTURE)
        self._tick_size_dict = DataFactory.get_tick_size_dict(symbols=list(self._symbols), asset_type=ASSETTYPE_FUTURE)

    def _get_history(self, startDate, endDate):
        self._market_data = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq=self._freq,
                                                          symbols=list(self._symbols), start_date=startDate,
                                                          end_date=endDate)

    def _get_history_daily(self, startDate, endDate):
        self._market_data_daily = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq='1d',
                                                                symbols=list(self._symbols), start_date=startDate,
                                                                end_date=endDate)

    def gen_signal(self, symbol, **kwargs):
        capital = kwargs['capital']

        ma_period_lst = self._params['ma_period'][symbol[:-4]]
        k_lst = self._params['k'][symbol[:-4]]
        # ma_period_lst = self._params['ma_period']
        # k_lst = self._params['k']
        contract_size_list = self._contract_size_dict[symbol]
        capital_intial = 10 * capital / len(self._symbol_pair_list) / contract_size_list
        pos_df_list = []
        signal_df_lst = []
        for i in range(len(ma_period_lst)):
            ma_period = ma_period_lst[i]
            k = k_lst[i]
            data = self._format_data(symbol, ma_period, k)
            _signal_lst = []
            _signal = 0
            for idx, _row in data.iterrows():
                condition_l = _row.close > _row.up
                condition_s = _row.close < _row.down
                if _signal == 0:
                    if condition_l:
                        cost = _row.close
                        _signal = _row.riskScaler * capital_intial / cost
                    elif condition_s:
                        cost = _row.close
                        _signal = -_row.riskScaler * capital_intial / cost

                elif _signal > 0:
                    if condition_s:
                        cost = _row.close
                        _signal = -_row.riskScaler * capital_intial / cost

                elif _signal < 0:
                    if condition_l:
                        cost = _row.close
                        _signal = _row.riskScaler * capital_intial / cost
                _signal_lst.append(_signal)
            data[symbol + str(i)] = _signal_lst
            data[symbol + str(i)] = data[symbol + str(i)].fillna(0)
            signal_df_lst.append(data[symbol + str(i)])
        signal_df = pd.concat(signal_df_lst, axis=1)
        signal = np.mean(signal_df, axis=1)

        target_pos_dict = {}
        target_pos_tmp = signal
        target_pos_tmp.fillna(method='pad', inplace=True)
        target_pos_tmp.name = symbol
        target_pos_dict[symbol] = target_pos_tmp
        if self.result_fold is not None:
            file_saver().save_file(target_pos_tmp, self.result_fold + '\\' + symbol + '_target_pos.csv')

        # 换月处理
        contract_id_series = data['contract_id']
        contract_id_series.name = symbol + '_contract_id'
        target_pos_df = pd.concat([target_pos_tmp, contract_id_series], axis=1)
        # target_pos_df.loc[
        #     target_pos_df[symbol + '_contract_id'] != target_pos_df[symbol + '_contract_id'].shift(-2), symbol] = 0
        target_pos_dict[symbol] = target_pos_df[symbol]

        pos_serires = target_pos_dict[symbol].copy()
        pos_serires.name = 'position'

        pos_df = pd.DataFrame(pos_serires, index=pos_serires.index)
        pos_df = pos_df.join(contract_id_series)
        pos_df['symbol'] = symbol
        pos_df['asset_type'] = self._asset_type
        pos_df['contract_size'] = self._contract_size_dict[symbol]
        pos_df['tick_size'] = self._tick_size_dict[symbol]
        pos_df['margin_ratio'] = 0.1
        pos_df['freq'] = self._freq
        pos_df['remark'] = '.'.join([self._strategy_name, self._strategy_type, symbol])

        pos_df_list.append(pos_df)
        signal_dataframe = None
        if len(pos_df_list) > 0:
            signal_dataframe = pd.concat(pos_df_list)
        return signal_dataframe

    @runing_time
    def _format_data(self, symbol, ma_period, k):
        targetVol = self._params['targetVol']
        volLookback = self._params['volLookback']

        data_daily = self._market_data_daily[symbol][
            ['high', 'close', 'open', 'low', 'volume', 'trade_date', 'contract_id', 'price_return']]
        data_daily = data_daily[data_daily['volume'] > 0]
        data_daily = data_daily \
                         .assign(ma=lambda df: talib.MA(df['close'].values, ma_period)) \
                         .assign(delta=lambda df: abs(df.close - df.close.shift(1))) \
                         .assign(delta_ma=lambda df: talib.MA(df['delta'].values, ma_period)) \
                         .assign(up=lambda df: df.ma + k * df.delta_ma) \
                         .assign(down=lambda df: df.ma - k * df.delta_ma)\
                         .loc[:, ['contract_id', 'up', 'down', 'high', 'low', 'close', 'open', 'trade_date', 'price_return']] \
                         .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)))

        if volLookback != 0:
            realizedVol = data_daily['price_return'].ewm(
                span=volLookback, ignore_na=True, adjust=False).std(bias=True) * (252 ** 0.5)
            # if symbol not in ['T_VOL', 'TF_VOL']:
            realizedVol = self.cap_vol_by_rolling(realizedVol, targetVol)
            riskScaler = targetVol / realizedVol
            data_daily['riskScaler'] = riskScaler * self.maxLeverage
        else:
            data_daily['riskScaler'] = self.maxLeverage
        return data_daily

    def run_test(self, startDate, endDate, **kwargs):
        self._get_market_info()
        self._get_history_daily(startDate=startDate, endDate=endDate)
        capital = kwargs['capital']
        signal_dataframe = []
        for symbol in self._symbols:
            signal_dataframe.append(
                self.gen_signal(symbol=symbol, capital=capital))
        signal_dataframe = pd.concat(signal_dataframe)
        # if self.result_fold is not None:
        #     signal_dataframe.to_csv(self.result_fold + '\\' + 'signal_dataframe.csv')
        return signal_dataframe