import logging

class Env(object):

    """Abstract class for an environment. Simplified OpenAI API.
    this gym provides environment for trading to agent


    ## reference
    keyword for trading http://www.fo24.co.kr/bbs/print_view.html?id=167497&code=bbs203
    doc http://docs.python-guide.org/en/latest/writing/documentation/
    """

    def _is_done(selfs):
        """
        check closing condition!!
            , lose all money
            . duration is exhausted..
            . exceed the count of available transaction
        :return:
        """
        pass


    def _get_status(self):
        """
        agent status information
        for now, we can only allow a agent to have only one ticker when its agent trades.

        :parameter
        . balance
        . num_equitiy_to_hold
        . remaining_count_to_buyhold
        .
        :return:
        """
        pass


    def __init__(self, buy_commission=0.015, hold_commision=0.315, start_money=1000000, max_count_buyhold=100, episode_duration_min = 390):
        """
        init (매수수수료=0.015, 매도수수료=0.315, 시작금액=1000000, 최대매매횟수=100, 매매시간(분) = 390)
        buy_commission=0.015, hold_commision=0.315, start_money=1000000, max_count_buyhold=100, episode_duration_min = 390
        """
        self.n_actions = None
        self.state_shape = None
        self.buy_commission = 0.015
        self.hold_commision = 0.315
        self.start_money = 1000000
        self.max_count_buyhold = 100
        self.episode_duration_min = 390


        #TODO : load all data of ticker from database into memory



    def _rewards(self):
        pass

    def _get_observation(self):
        """ we find data of current timestamp

        bid / ask quote top 3
        open high close 3 closest from present price
        low
        spot price : price in present
        percentage : increase or decrease percentage compared to its close price of yesterday
        cumulated_volume
        cumulated_transaction_money
        mean_transaction_cost
        strongnessTosucceed
        transaction_money_by_unit_time
        volume_by_unit_time
        time : it is important feature which means timestamp that environment has when agent requested

        ## consideration !!!
        we don't consider any values that agent only knows : balance, the remaining_order_count
        """
        return []


    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (numpy.array): action array
        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (str): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
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
        """Render the environment.
        """
        logging.info(self._get_status())
