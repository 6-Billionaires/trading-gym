from gym.core import Env
from collections import deque
from  util.Exception import TradingException

import pandas as pd
import numpy as np
import glob
import random
import datetime
import math
import config
import os
import pickle
import os.path
import logging
import time
from gym_core import ioutil

#logging.basicConfig(filename='trading-gym-{}.log'.format(time.strftime('%Y%m%d%H%M%S')),level=logging.DEBUG)
import logging

logging.basicConfig(
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(config.GYM['HOME'], 'trading-gym-{}.log'
                                                       .format(time.strftime('%Y%m%d%H%M%S')))),
        logging.StreamHandler()
    ])


c_home_full_dir = config.GYM['HOME']

class TradingGymEnv(Env):

    d_episodes_data = {}
    l_err_occurred_idx = []

    # every agent has their own constraints to be managed by trading gym itself.
    # c_agent_max_num_of_allowed_transaction = 10
    c_pickle_data_file = 'tgym-data.pickle'

    c_step_start_yyyymmdd_in_episode = '090601'  # 09:06:01
    c_prev_step_start_datetime_in_episode = None
    c_step_start_datetime_in_episode = None
    c_range_timestamp = []

    p_current_num_transaction = 0
    p_current_episode_ref_idx = 0
    p_current_episode_data_quote = None
    p_current_episode_data_order = None
    p_current_episode_ticker = None
    p_current_episode_date = None
    p_current_episode_price_history = deque(maxlen=60)
    p_current_step_in_episode = 0 # step interval = 1sec
    p_max_num_of_allowed_transaction = 10
    p_is_stop_loss = None
    p_is_reached_goal = None

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
        if self.p_current_num_transaction >= self.p_max_num_of_allowed_transaction:
            return True
        elif self.p_current_step_in_episode >= self.episode_duration_min * 60:
            return True
        elif self.p_is_stop_loss:
            return False
        else:
            return False

    def create_episode_data(self, episode_type, episode_count=1):
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
        # Loading pickle file
        if os.path.isfile(ioutil.make_dir(c_home_full_dir, self.c_pickle_data_file)):
            logging.debug('It is loading pickle file instead of reading csv files for performance.')

            self.d_episodes_data = pickle.load(open(c_home_full_dir + '/' + self.c_pickle_data_file, "rb"))
            logging.debug('it loaded pickle file without reading csv file' +
                          'of equities and loaded {} equities information.'.format(len(self.d_episodes_data)))
        else:
            logging.debug('There is no pickle file so that we are starting reading csv files.')
            self.d_episodes_data = ioutil.load_data_from_directory(c_home_full_dir, '0', episode_count)
            logging.debug('we are saving pickle file not to load files again..')
            pickle.dump(self.d_episodes_data, open(c_home_full_dir + '/' + self.c_pickle_data_file, "wb"))
        return len(self.d_episodes_data)

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
            self.episode_data_count = self.create_episode_data(episode_type, 1000)

        # for now, episode type is not considered.
        # self.p_current_episode_ref_idx = random.randint(0, self.episode_data_count - 1)

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

        p_current_episode_price_history = deque(maxlen=self.episode_duration_min)

        self.p_current_step_in_episode = 0
        self.p_max_num_of_allowed_transaction = max_num_of_transaction
        self.p_is_stop_loss_price = 0

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
        p0 = self.d_episodes_data[self.p_current_episode_ref_idx]['quote'].loc[self.c_range_timestamp[self.p_current_step_in_episode]]
        p1 = self.d_episodes_data[self.p_current_episode_ref_idx]['order'].loc[self.c_range_timestamp[self.p_current_step_in_episode]]
        p2 = self.p_current_episode_price_history

        return self.observation_processor(np.append(np.append(p0, p1), p2[-1]))

    def observation_processor(self, observation):
        return observation

    def step(self, action):
        """
        Here is the interface to be called by its agent.
        _get_observation needs to be transformed using transform observation that __init__ received.

        Args:
            action (object):
                1  : buy a stock with all money I have
                -1 : sell every stock I have


        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if action == 0:

            self.p_current_step_in_episode += 1
            next_base_price = self.p_current_episode_data_order.loc[
                self.c_range_timestamp[self.p_current_step_in_episode]]['SellHoga1']

            if next_base_price <= -100:
                can_buy = False
            else:
                can_buy = True

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

        else:
            # base_price is the price that agent can buy the stock at now.
            base_price = self.p_current_episode_data_order.loc[
                self.c_range_timestamp[self.p_current_step_in_episode]]['SellHoga1']

            best_price = -10000000000
            self.p_is_stop_loss = False
            self.p_is_reached_goal = False
            buy_price = -1
            last_price = -1

            length = len(pd.date_range(
                    self.c_range_timestamp[self.p_current_step_in_episode],
                    self.c_range_timestamp[
                        np.minimum(self.p_current_step_in_episode + 60, self.c_episode_max_step_count - 1)], freq='S'
            ))

            for idx, present_ts in enumerate(pd.date_range(
                    self.c_range_timestamp[self.p_current_step_in_episode],
                    self.c_range_timestamp[
                        np.minimum(self.p_current_step_in_episode + 60, self.c_episode_max_step_count - 1)], freq='S'

            )):

                present_price = self.p_current_episode_data_order.loc[present_ts]['BuyHoga1']

                percent = ((present_price+100) - (base_price+100)) / ( base_price+100) * 100

                if idx == 0:
                    buy_price = self.p_current_episode_data_order.loc[present_ts]['SellHoga1']
                if idx == length - 1:  # if you change the window size, you must change it.
                    last_price = self.p_current_episode_data_order.loc[present_ts]['BuyHoga1']

                if not self.p_is_reached_goal and percent < 0 and self.percent_stop_loss <= np.abs(percent):
                    self.p_is_stop_loss = True
                    self.p_is_stop_loss_price = present_price  # TODO: ASK1 is correct price for stop loss ?!
                    break
                elif percent > 0 and self.percent_goal_profit <= np.abs(percent):
                    self.p_is_reached_goal = True
                    if best_price < present_price:
                        best_price = present_price
                else:
                    pass

            # adding one step after all processes are done.
            self.p_current_step_in_episode = self.p_current_step_in_episode + 1
            next_base_price = self.p_current_episode_data_order.loc[
                self.c_range_timestamp[self.p_current_step_in_episode]]['SellHoga1']
            if next_base_price <= -100:
                can_buy = False
            else:
                can_buy = True

            _info = {}
            _info['stop_loss'] = self.p_is_stop_loss
            _info['stop_loss_price'] = self.p_is_stop_loss_price
            _info['reached_profit'] = self.p_is_reached_goal
            _info['best_price'] = best_price
            _info['can_buy'] = can_buy
            _info['buy_price'] = buy_price
            _info['last_price'] = last_price

            _observation = self._get_observation()
            _done = self._is_done()

            return _observation, self._rewards(_observation, action, _done, _info), _done, _info

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        here we will reset agent's environment to restart.
        for now, we need to have its database to load and the status will go back to its first status like it called init

        every time, it needs to reset index to reference which index of episode data agent use training data.

        Returns:
            numpy.array: The initial observation of the space. Initial reward is assumed to be 0.
        """
        # TODO : I can't not remember why I think it needs error exception here.
        try:
            self.p_current_episode_ref_idx = random.randint(0, self.episode_data_count - 1)
            self.p_current_step_in_episode = 0

            self.p_current_episode_ticker = self.d_episodes_data[self.p_current_episode_ref_idx]['meta']['ticker']
            self.p_current_episode_date = self.d_episodes_data[self.p_current_episode_ref_idx]['meta']['date']
            self.p_current_episode_data_quote = self.d_episodes_data[self.p_current_episode_ref_idx]['quote']
            self.p_current_episode_data_order = self.d_episodes_data[self.p_current_episode_ref_idx]['order']

            current_date = self.p_current_episode_date

            logging.debug('this episode start with ticker {} and date {}',format(self.p_current_episode_ticker,self.p_current_episode_date))

            # it can not declared in init method. because we need to consider yyyymmdd in future.
            self.c_prev_step_start_datetime_in_episode = datetime.datetime(int(current_date[0:4]),
                                                                           int(current_date[4:6]),
                                                                           int(current_date[6:8]), 9, 5)
            self.c_step_start_datetime_in_episode = datetime.datetime(int(current_date[0:4]),
                                                                      int(current_date[4:6]),
                                                                      int(current_date[6:8]), 9, 6)

            self.c_range_timestamp = pd.date_range(
                self.c_step_start_datetime_in_episode, periods=self.c_episode_max_step_count + 1, freq='S')

            start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 9, 5)
            prev_read_rng = \
                pd.date_range(start, periods=np.minimum(
                    self.c_episode_max_step_count-self.p_current_step_in_episode,
                    self.episode_duration_min*60), freq='S')

            for prev_time_step in prev_read_rng:
                self.p_current_episode_price_history.append(
                        self.p_current_episode_data_quote.loc[prev_time_step]['Price(last excuted)'])
        except Exception:
            print('we get error')


        return self._get_observation()

    def render(self, mode=None):
        """
        Render the environment.
        display agent and gym's status based on a user configures
        """
        logging.info(self._get_status())

    def _get_status(self):
        logs = {}
        logs['current_num_transaction'] = self.p_current_num_transaction
        return logs