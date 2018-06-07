import logging
from gym.core import Env
import pandas as pd
import numpy as np
import glob
from  util.Exception import TradingException
import random
import datetime
import math
import config
import os
from collections import deque
import pickle
import os.path
import logging
import time

logging.basicConfig(filename='trading-gym-{}.log'.format(time.strftime('%Y%m%d%H%M%S')),level=logging.DEBUG)
C_HOME_FULL_DIR = config.GYM['HOME']


class TradingGymEnv(Env):

    d_episodes_data = {}

    # every agent has their own constraints to be managed by trading gym itself.
    c_agent_max_num_of_allowed_transaction = 10
    c_pickle_data_file = 'tgym-data.pickle'

    # c_agent_prev_step_start_yyyymmdd_in_episode = '090501'  # 9hr 6min 1sec
    # c_agent_step_start_yyyymmdd_in_episode = '090601' # 9hr 6min 1sec

    c_agent_prev_step_start_datetime_in_episode = None
    c_agent_step_start_datetime_in_episode = None

    c_agent_range_timestamp = []

    p_agent_current_num_transaction = 0
    p_agent_current_episode_ref_idx = 0
    p_agent_current_episode_data_quote = None
    p_agent_current_episode_data_order = None
    p_agent_current_episode_ticker = None
    p_agent_current_episode_date = None
    p_agent_current_episode_price_history = deque(maxlen=60)
    p_agent_current_step_in_episode = 0 # step interval = 1sec
    p_agent_max_num_of_allowed_transaction = 10

    p_agent_is_stop_loss = None
    p_agent_is_reached_goal = None

    is_data_loaded = False
    episode_data_count = 0
    episode_duration_min = 60

    """
    This class's super class is from OpenAI Gym and extends to provide trading environment

    # reference
        - keyword for trading
            . http://www.fo24.co.kr/bbs/print_view.html?id=167497&code=bbs203
        - doc
            . http://docs.python-guide.org/en/latest/writing/documentation/
        - style guide to write comment ( or docstring)
            . https://medium.com/@kkweon/%ED%8C%8C%EC%9D%B4%EC%8D%AC-doc-%EC%8A%A4%ED%83%80%EC%9D%BC-%EA%B0%80%EC%9D%B4%EB%93%9C%EC%97%90-%EB%8C%80%ED%95%9C-%EC%A0%95%EB%A6%AC-b6d27cd0a27c
            . http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
    """
    def _is_done(self):
        """
        check whether or not this episode is ended.

        Environment side
        ========================
        . current step exceeds in max_steps
        . exceed episode time bound

        Agent side
        ========================
        . lose all money agent has (= no balance)
        . exceed the count of available transaction

        :return: True or False

        """
        # TODO : here we need to decide whether or not episode ends right way or wait until end of duration of original duration
        if self.p_agent_current_num_transaction >= self.p_agent_max_num_of_allowed_transaction:
            return True
        elif self.p_agent_current_step_in_episode >= self.episode_duration_min * 60:
            return True
        elif self.p_agent_is_stop_loss:
            return False
        else:
            return False

    def create_episode_data(self, episode_type):

        """
            it is executed just one time right after this gym is made for the first time.

            # 1. Loading data into gym
                It loads csv files based on its episode type. Before, we need to ensure that those csv files have to locate in a
                directory where trading-episode-filter gathered according to rules of episode type

            # 2. the file to be loaded
                - data/episode_type/ticker/episode_type-ticker-yyyymmdd.csv

            # 3. csv format is look like below
                episode_type-AAPL-yyyymmdd-quote.csv,
                episode_type-AAPL-yyyymmdd-order.csv
        """
        _idx = 0

        if os.path.isfile(C_HOME_FULL_DIR + '/' + self.c_pickle_data_file):
            logging.debug('It is loading pickle file instead of reading csv files for performance.')
            d_episodes_data = pickle.load(open(C_HOME_FULL_DIR + '/' + self.c_pickle_data_file, "rb"))
            len(self.d_episodes_data.keys())

        logging.debug('There is no pickle file so that we are starting reading csv files.')

        for item in glob.glob(C_HOME_FULL_DIR + '/data/' + episode_type + '/*'):
            d_order = pd.DataFrame()
            d_quote = pd.DataFrame()
            d_meta = {}

            for pf in glob.glob(item + '/*-order.csv', ):

                try:

                    f = pf.split('\\')[-1]
                    f = f.split('/')[-1]

                    """
                    1. condition
                        for now, there is only one episode type. 
                         - "episode_type" : 0
    
                    2. data shape
                        {
                            "episode_idx" : "episode_data" : {
                                                                    "meta" : {
                                                                        "ticker" : "AAPL",
                                                                        "date"   : "20180501"
                                                                    },
                                                                    "quote" : dataframe generated by pandas after reding csv, 
                                                                    "order" : dataframe generated by pandas after reding csv,
                                                                    }
                                                             }
                    """

                    current_ticker = f.split('-')[1]
                    current_date = f.split('-')[2]

                    d_meta = {'ticker': current_ticker, "date": current_date}  # 1


                    if os.path.sep == '\\' or '/':
                        d_order = pd.read_csv(item+'/'+f, index_col=0, parse_dates=True)  # 2
                    else:
                        d_order = pd.read_csv(f, index_col=0, parse_dates=True)  # 2

                    #
                    quote_f = episode_type + '-' + current_ticker + '-' + current_date + '-quote.csv'
                    if os.path.sep == '\\' or '/':
                        d_quote = pd.read_csv(item+'/'+quote_f, index_col=0, parse_dates=True)  # 3
                    else:
                        d_quote = pd.read_csv(quote_f, index_col=0, parse_dates=True)  # 3


                    # # TODO : delete below. it is just fake data of two above
                    # d_order = pd.DataFrame(np.random.randn(3600, 20),
                    #                        columns=['ask_price1', 'ask_price2', 'ask_price3', 'ask_price4', 'ask_price5',
                    #                                 'ask_price6', 'ask_price7', 'ask_price8', 'ask_price9', 'ask_price10',
                    #                                 'bid_price1', 'bid_price2', 'bid_price3', 'bid_price4', 'bid_price5',
                    #                                 'bid_price6', 'bid_price7', 'bid_price8', 'bid_price9', 'bid_price10',
                    #                                 ],
                    #                        index=pd.date_range(start='1/1/2016', periods=60 * 60, freq='S'))
                    #
                    # d_quote = pd.DataFrame(np.random.randn(3600, 11),
                    #                        columns=['buy_amount', 'buy_weighted_price', 'sell_amount',
                    #                                 'sell_weighted_price', 'total_amount', 'total_weighted_price',
                    #                                 'executed_strength', 'open', 'high', 'low', 'present_price'],
                    #                        index=pd.date_range(start='1/1/2016', periods=60 * 60, freq='S'))

                except (KeyError, RuntimeError, TypeError, NameError, ValueError, Exception):
                    print('We got error in {}'.format(f))

            d_episode_data = {}
            d_episode_data['meta'] = d_meta
            d_episode_data['quote'] = d_quote
            d_episode_data['order'] = d_order

            self.d_episodes_data[_idx] = d_episode_data
            _idx = _idx + 1

        logging.debug('we are saving pickle file not to load files again..')
        pickle.dump(self.d_episodes_data, open(C_HOME_FULL_DIR + '/' + self.c_pickle_data_file, "wb"))

        return len(self.d_episodes_data.keys())

    def __init__(self, episode_type=None, episode_duration_min = 60, step_interval='1s', percent_stop_loss=10, percent_goal_profit = 2,
                 balanace = None, max_num_of_transaction=10, obs_transform=None):
        """
        Initialize environment

        Args:
            episode type
                0 : treat equity which reached upper limit yesterday ( in development )
                1 : treat equity which start 5% higher price compared to close price yesterday ( not development yet )
            episode_duration_min
                this parameter will limit episode time
            step_interval
                this value is consist of number and unit of time ('s' for second, 'm' for minute, 'h' for hour ,'d' for day)
            percent_stop_loss
                positive. percentage, if a equity which agent holds fall down down to its percentage, it will be sold automatically.
            percent_goal_profit
                positive. percentage, if action from step results in this profit within duration, this action is considered as good and agent would get good reward from this action.
        """
        # self.episode_data_count = 0
        if not self.is_data_loaded:
            self.episode_data_count = self.create_episode_data(episode_type)

        # for now, episode type is not considered.
        #self.p_agent_current_episode_ref_idx = random.randint(0, self.episode_data_count - 1)

        self.c_episode_max_step_count = 60 * episode_duration_min

        self.reset()

        # this parameter is belong to episode type so that it isn't necessary anymore.
        self.percent_stop_loss = percent_stop_loss
        self.percent_goal_profit = percent_goal_profit
        self.n_actions = None
        self.state_shape = None
        self.interval = 1   # 1 second
        self.ob_transform = obs_transform
        self.episode_duration_min = episode_duration_min

        p_agent_current_episode_price_history = deque(maxlen=self.episode_duration_min)

        self.p_agent_current_step_in_episode = 0
        self.p_agent_max_num_of_allowed_transaction = max_num_of_transaction
        self.p_agent_is_stop_loss_price = 0


    def init_observation(self):
        return self._get_observation()

    def _rewards(self, observation, action, done, info):
        """
        for now, reward is just additional information other than observation itself.
        @return: the price of the current episode's equity's price 60 secs ahead
        """
        raise NotImplementedError

    def _get_observation(self):
        """
        Obesrvation information
        ===============================
        0. time : yyyymmdd
        1. bid / ask quote top 10
        2. open high close low based on step_interval in env
        3. price history for 1 minute
        """
        p0 = self.d_episodes_data[self.p_agent_current_episode_ref_idx]['quote'].loc[self.c_agent_range_timestamp[self.p_agent_current_step_in_episode]]
        p1 = self.d_episodes_data[self.p_agent_current_episode_ref_idx]['order'].loc[self.c_agent_range_timestamp[self.p_agent_current_step_in_episode]]
        p2 = self.p_agent_current_episode_price_history

        return np.append(np.append(p0, p1), p2)

    def step(self, action):
        """
        Here is the interface to be called by its agent.
        _get_observation needs to be transformed using transform observation that __init__ received.

        Args:
            action (object): for now, action is simple! 1 or -1 ( buy all or sell all )
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if action == 0:
            self.p_agent_current_step_in_episode = self.p_agent_current_step_in_episode + 1
            next_base_price = self.p_agent_current_episode_data_order.loc[
                self.c_agent_range_timestamp[self.p_agent_current_step_in_episode]]['SellHoga1']
            if next_base_price <= -100:
                can_buy = False
            else:
                can_buy = True

            # _info = [{'stop_loss': False,
            #             'stop_loss_price': -1,
            #             'reached_profit': False,
            #             'best_price': -1,
            #             'can_buy': can_buy}]
            _info = {}
            _info['stop_loss'] = False
            _info['stop_loss_price'] = -1
            _info['reached_profit'] = False
            _info['best_price'] = -1
            _info['can_buy'] = can_buy
            _info['buy_price'] = -1
            _info['last_price'] = -1

            _observation = self._get_observation()
            _done = self._is_done()

            return _observation, self._rewards(_observation, action, _done, _info), _done, _info
        else :

            # base_price is the price that agent can buy the stock at now.
            base_price = self.p_agent_current_episode_data_order.loc[
                self.c_agent_range_timestamp[self.p_agent_current_step_in_episode]]['SellHoga1']

            best_price = -10000000000
            self.p_agent_is_stop_loss = False
            self.p_agent_is_reached_goal = False
            buy_price = -1
            last_price = -1

            length = len(pd.date_range(
                    self.c_agent_range_timestamp[self.p_agent_current_step_in_episode],
                    self.c_agent_range_timestamp[
                        np.minimum(self.p_agent_current_step_in_episode+60, self.c_episode_max_step_count-1)], freq='S'
            ))

            for idx, present_ts in enumerate(pd.date_range(
                    self.c_agent_range_timestamp[self.p_agent_current_step_in_episode],
                    self.c_agent_range_timestamp[
                        np.minimum(self.p_agent_current_step_in_episode+60, self.c_episode_max_step_count-1)], freq='S'

            )):

                present_price = self.p_agent_current_episode_data_order.loc[present_ts]['BuyHoga1']

                percent = ((present_price+100) - (base_price+100)) / ( base_price+100) * 100

                if idx == 0:
                    buy_price = self.p_agent_current_episode_data_order.loc[present_ts]['SellHoga1']
                if idx == length - 1:  # if you change the window size, you must change it.
                    last_price = self.p_agent_current_episode_data_order.loc[present_ts]['BuyHoga1']

                if not self.p_agent_is_reached_goal and percent < 0 and self.percent_stop_loss <= np.abs(percent):
                    self.p_agent_is_stop_loss = True
                    self.p_agent_is_stop_loss_price = present_price  # TODO: ASK1 is correct price for stop loss ?!
                    break
                elif percent > 0 and self.percent_goal_profit <= np.abs(percent):
                    self.p_agent_is_reached_goal = True
                    if best_price < present_price:
                        best_price = present_price
                else:
                    pass

            # adding one step after all processes are done.
            self.p_agent_current_step_in_episode = self.p_agent_current_step_in_episode + 1
            next_base_price = self.p_agent_current_episode_data_order.loc[
                self.c_agent_range_timestamp[self.p_agent_current_step_in_episode]]['SellHoga1']
            if next_base_price <= -100:
                can_buy = False
            else:
                can_buy = True

            _info = {}
            _info['stop_loss'] = self.p_agent_is_stop_loss
            _info['stop_loss_price'] = self.p_agent_is_stop_loss_price
            _info['reached_profit'] = self.p_agent_is_reached_goal
            _info['best_price'] = best_price
            _info['can_buy'] = can_buy
            _info['buy_price'] = buy_price
            _info['last_price'] = last_price

            _observation = self._get_observation()
            _done = self._is_done()

            return _observation, self._rewards(_observation, action, _done, _info), _done, _info


    def reset(self):
        """Reset the state of the environment and returns an initial observation.

        here we will reset agent's enviroment to restart.
        for now, we need to have its database to load and the status will go back to its first status like it called init

        everytime, it needs to reset index to reference which index of episode data agent use training data.

        Returns:
            numpy.array: The initial observation of the space. Initial reward is assumed to be 0.
        """
        self.p_agent_current_episode_ref_idx = random.randint(0, self.episode_data_count-1)
        self.p_agent_current_step_in_episode = 0

        self.p_agent_current_episode_ticker = self.d_episodes_data[self.p_agent_current_episode_ref_idx]['meta']['ticker']
        self.p_agent_current_episode_date = self.d_episodes_data[self.p_agent_current_episode_ref_idx]['meta']['date']
        self.p_agent_current_episode_data_quote = self.d_episodes_data[self.p_agent_current_episode_ref_idx]['quote']
        self.p_agent_current_episode_data_order = self.d_episodes_data[self.p_agent_current_episode_ref_idx]['order']

        current_date = self.p_agent_current_episode_date

        print(self.p_agent_current_episode_ticker)
        print(self.p_agent_current_episode_date)

        # it can not declared in init method. because we need to consider yyyymmdd in future.
        self.c_agent_prev_step_start_datetime_in_episode = datetime.datetime(int(current_date[0:4]),
                                                                             int(current_date[4:6]),
                                                                             int(current_date[6:8]), 9, 5)
        self.c_agent_step_start_datetime_in_episode = datetime.datetime(int(current_date[0:4]),
                                                                        int(current_date[4:6]),
                                                                        int(current_date[6:8]), 9, 6)

        self.c_agent_range_timestamp = pd.date_range(
            self.c_agent_step_start_datetime_in_episode, periods=self.c_episode_max_step_count+1, freq='S')

        start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 9, 5)
        prev_read_rng = \
            pd.date_range(start, periods=np.minimum(
                self.c_episode_max_step_count-self.p_agent_current_step_in_episode,
                self.episode_duration_min*60), freq='S')

        for prev_time_step in prev_read_rng:
            self.p_agent_current_episode_price_history.append(
                    self.p_agent_current_episode_data_quote.loc[prev_time_step]['Price(last excuted)'])

        return self._get_observation()

    def render(self, mode=None):
        """
        Render the environment.
        display agent and gym's status based on a user configures
        """
        logging.info(self._get_status())

    def _get_status(self):
        logs = {}
        logs['agent_current_num_transaction'] = self.p_agent_current_num_transaction
        return logs


