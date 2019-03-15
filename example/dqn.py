from core.tgym import TradingGymEnv as env
from collections import deque
import numpy as np

from keras.models import Model
from keras.layers import LeakyReLU, Input, Dense, Conv3D, Conv1D, Dense, \
    Flatten, MaxPooling1D, MaxPooling2D, MaxPooling3D, Concatenate
from keras.optimizers import Adam


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from example.agent import DQNAgent

class ENV(env):

    # data shape
    rows = 10
    columns = 2
    seconds = 60
    channels = 2
    features = 11

    # def __init__(self, **kwargs):
    #     super(ENV, self).__init__(**kwargs)


    def _rewards(self, observation, action, done, info):
        # print('observation : {}'.format(observation))
        # print('action : {}'.format(action))
        return 0

    holder_observation = deque(np.array([[0 for x in range(52)] for y in range(seconds)]), maxlen=seconds)


env = ENV(episode_type='0', percent_goal_profit=2, percent_stop_loss=5, episode_duration_min=60)

WINDOW_LENGTH = 4
EPISODES = 10000


def build_network_for_sparsed(optimizer='adam',init_mode='uniform', filters=16, neurons=20, activation='relu'):
    if activation == 'leaky_relu':
        activation = LeakyReLU(alpha=0.3)

    input_order = Input(shape=(10, 2, 60, 2), name="x1")
    input_tranx = Input(shape=(60, 11), name="x2")

    h_conv1d_2 = Conv1D(filters=16, kernel_initializer=init_mode, kernel_size=3)(input_tranx)
    h_conv1d_2 = LeakyReLU(alpha=0.3)(h_conv1d_2)
    h_conv1d_4 = MaxPooling1D(pool_size=3,  strides=None, padding='valid')(h_conv1d_2)
    h_conv1d_6 = Conv1D(filters=32, kernel_initializer=init_mode, kernel_size=3)(h_conv1d_4)
    h_conv1d_6 = LeakyReLU(alpha=0.3)(h_conv1d_6)
    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(filters=filters, kernel_initializer=init_mode, kernel_size=(2, 1, 5))(input_order)
    h_conv3d_1_1 = LeakyReLU(alpha=0.3)(h_conv3d_1_1)
    h_conv3d_1_2 = Conv3D(filters=filters,  kernel_initializer=init_mode,kernel_size=(1, 2, 5))(input_order)
    h_conv3d_1_2 = LeakyReLU(alpha=0.3)(h_conv3d_1_2)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(kernel_initializer=init_mode, filters=filters*2, kernel_size=(1, 2, 5))(h_conv3d_1_3)
    h_conv3d_1_5 = LeakyReLU(alpha=0.3)(h_conv3d_1_5)

    h_conv3d_1_6 = Conv3D(kernel_initializer=init_mode, filters=filters*2, kernel_size=(2, 1, 5))(h_conv3d_1_4)
    h_conv3d_1_6 = LeakyReLU(alpha=0.3)(h_conv3d_1_6)

    h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
    h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
    o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

    o_conv3d_1_1 = Flatten()(o_conv3d_1)

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1])

    i_concatenated_all_h = Dense(neurons, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    output = Dense(2, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx], output)
    # model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])

    # model.summary()

    return model


if __name__ == "__main__":

    training_mode = True

    state_size = env.observation_space.n
    action_size = env.action_space.n

    print('state_size', state_size, 'action_size', action_size)

    agent = DQNAgent(state_size, action_size, build_model=build_network_for_sparsed, model_file_path="./save_model/"
                     , model_filename="model", model_file_ext=".h5", load_model=True)

    scores, steps, hps, episodes = [], [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        my_hp = 0
        step_cnt = 0
        kill = False
        # env 초기화
        state = env.reset()

        while not done:
            action = agent.get_action(state, training_mode)
            next_state, reward, done, info = env.step(action)

            # reward design

            if training_mode:
                agent.append_sample(state, action, reward, next_state, done)
                if len(agent.memory) >= agent.train_start:
                    agent.train_model()

            score += reward
            state = next_state

        print("episode: {:-5}  reward: {:-7}  memory length: {:-4}  epsilon: {:<20}".format(e, score, len(agent.memory),agent.epsilon))

        if training_mode:
            agent.update_target_model()

    env.close()
