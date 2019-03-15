import os
import random
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, build_model, load_model=True, discount_factor=0.99, learning_rate=0.001
                , epsilon_start=1.0, epsilon_decay=0.9, exploration_steps=None, epsilon_min=0.01, batch_size=64, train_start=1000
                , max_queue_size=8000, max_tmp_queue_size=10, model_file_path='', model_filename='', model_file_ext='',
                 enable_dueling=False, enable_double=False, enable_drqn=False):

        self.enable_drqn = enable_drqn
        self.enable_dueling = enable_dueling
        self.enable_double = enable_double

        self.render = False
        self.load_model = load_model

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.exploration_steps = exploration_steps
        if self.exploration_steps is None:
            self.epsilon_decay = epsilon_decay
        else:
            self.epsilon_decay = (epsilon_start - self.epsilon_min) / self.exploration_steps
        self.batch_size = batch_size
        self.train_start = train_start

        self.memory = deque(maxlen=max_queue_size)
        self.tmp_memory = deque(maxlen=max_tmp_queue_size)

        self.model = build_model(self)
        self.target_model = build_model(self)

        self.update_target_model()
        self.model_file_path = model_file_path
        self.model_file_name = model_filename
        self.model_file_ext = model_file_ext
        self.model_file_pull_path = model_file_path + model_filename + model_file_ext

        if self.load_model and os.path.isfile(self.model_file_pull_path):
            self.model.load_weights(self.model_file_pull_path)
            self.epsilon = self.epsilon_min

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, training_mode):
        if training_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            if self.enable_drqn:
                q_value = self.model.predict([np.expand_dims(state[0],0), np.expand_dims(state[1],0)])
            else:
                q_value = self.model.predict([np.expand_dims(state[0],0), np.expand_dims(state[1],0)])
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.tmp_memory.append([state, action, reward, next_state, done])

    def train_model(self):
        if self.epsilon > self.epsilon_min and len(self.memory) >= self.train_start:
            if self.exploration_steps is None:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon -= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        if self.enable_drqn:
            states = np.zeros((self.batch_size, self.state_size[0], self.state_size[1]))
            next_states = np.zeros((self.batch_size, self.state_size[0],self.state_size[1]))
        else:
            states = np.zeros((self.batch_size, self.state_size))
            next_states = np.zeros((self.batch_size, self.state_size))

        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model.predict(states)

        target_val = 0

        target_val = self.target_model.predict(next_states)

        if self.enable_double:
            origin_val = self.model.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                if self.enable_double:
                    if np.random.random() < 0.5:
                        target[i][actions[i]] = rewards[i] + self.discount_factor * target_val[i][np.argmax(origin_val[i])]
                    else:
                        target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])
                else:
                    target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)

