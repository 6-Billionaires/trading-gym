# Trading Gym for Quantitative trading


## Intro

This is a trading gym for any agent to trade for short term trading. We have enermous data for short term trading. We have been gathering for every Korean equities order and quote data every tick moment and also reflected data to our trading gym. In this environment, you can testify your own agent which beat market and results in making you rich someday.  

## Architecture
It's is simple architecture that you motivate follow and run this repo easily.

![](/materials/architecture.png)

## Agent with this gym
[Here](https://github.com/6-Billionaires/trading-agent) can be reference. Those are that we built in temporary.


## How to run
You can clone two repository into your local computer or cloud whatever. And you can run agent after including traidng gym as submodule or add gym's root into Python Path and then you can run trading agent itself. so simple!



## Reference

### Other Gyms already built
1. Trading gym followed by OpenAI Gym architecture, spread trading
- https://github.com/Prediction-Machines/Trading-Gym

2. Trading gym followed by OpenAI Gym architecture, easy to look around with ipython example
- https://github.com/hackthemarket/gym-trading
- https://github.com/hackthemarket/gym-trading/blob/master/gym_trading/envs/TradingEnv.ipynb

3. Sairen
- Trading gym using API of Interative Broker  
- It is such a good reference for us. we will adapt live/paper mode feature of it.
    - https://doctorj.gitlab.io/sairen/
    - edemo / demouser
    * TWS configuration
        . TWS session will be set for socket port 7496 (live),
        . a paper account session will listen on socket port 7497 (paper)
        . https://interactivebrokers.github.io/tws-api/initial_setup.html#gsc.tab=0
4. deep trader
- [link](https://github.com/deependersingla/deep_trader)

### Reinforcement learning
[Algorithms collections based on OpenAI Gym](https://github.com/rll/rllab)
[best article](https://discuss.openai.com/t/gym-and-agent-for-algorithmic-stock-and-cryptocurrency-trading/519/23) to make trading gym on Discourse

### Plan

1. Packaging  
[setup.py](https://www.digitalocean.com/community/tutorials/how-to-write-modules-in-python-3#accessing-modules-from-another-directory), [Upload package into pip repo ](https://stackoverflow.com/questions/15746675/how-to-write-a-python-module-package)
2. Refactoring
[1](http://docs.python-guide.org/en/latest/writing/structure/), [2](https://jeffknupp.com/blog/2014/02/04/starting-a-python-project-the-right-way/)

2. Run this on cloud and allow every agent can access through REST API to train
