import logging
from gym.core import Env


class TradingGymEnv(Env):

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

        Agent side
        ========================
        . lose all money agent has (= no balance)
        . exceed the count of available transaction

        :return: True or False

        """
        pass


    def __init__(self, episode_duration_min = 390, obs_transform=None):
        """
        episode_duration_min = 390
        """
        self.n_actions = None
        self.state_shape = None
        self.episode_duration_min = 390
        self.interval = 1   # 1 second
        self.ob_transform = obs_transform

        #TODO : load all data of ticker from database into memory


    def _rewards(self):
        pass

    def _get_observation(self):
        """ we find data of current timestamp

        bid / ask quote top 10
        open high close low based on interval in env
        # 3 closest from present price
        spot price : price in present
        # percentage : increase or decrease percentage compared to its close price of yesterday

        # cumulated_volume
        # cumulated_transaction_money

        # mean_transaction_cost

        # strongnessTosucceed

        # transaction_money_by_unit_time
        # volume_by_unit_time
        time : it is important feature which means timestamp that environment has when agent requested

        ## consideration !!!
        we don't consider any values that agent only knows : balance, the remaining_order_count
        """
        return []


    def step(self, action):

        """
        action :
        for now, we just make action -1 or 1 ( buy all or sell all!! )

        _get_observation needs to be transformed using transform observation that __init__ received.
        """


        raise NotImplementedError()

    def reset(self):
        """Reset the state of the environment and returns an initial observation.

        here we will reset agent's enviroment to restart.
        for now, we need to have its database to load and the status will go back to its first status like it called init


        Returns:
            numpy.array: The initial observation of the space. Initial reward is assumed to be 0.
        """
        raise NotImplementedError()


    def render(self):
        """
        Render the environment.
        Need to display based on rules a user configures
        """
        logging.info(self._get_status())
        pass


