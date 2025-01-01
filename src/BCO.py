import gym
import numpy as np
import torch
import pygame
print("Pygame installed successfully!")

from src.NN import NN


class BCO:
    def __init__(self, hyperparameter_list, env: gym.Env, demoer):
        self.hyperparameter_list = hyperparameter_list

        self.world_model = self.get_model(hyperparameter_list)
        self.BC = self.get_policy(hyperparameter_list)
        self.world_model.init_model()
        self.BC.init_model()
        self.num_it = hyperparameter_list['num_iteration']
        self.rep = hyperparameter_list['rep']
        self.env = env
        self.demoer = demoer

    def get_model(self, hyperparameter):
        return NN(
            num_hidden_layer=hyperparameter['num_hidden_layer'],
            num_hidden_node=hyperparameter['num_hidden_node'],
            input_shape=hyperparameter['model_shape'],
            activation=hyperparameter['activation'],
            output_size=hyperparameter['output_size'],
            output_activation=hyperparameter['output_activation'],
            optimizer=hyperparameter['optimizer'],
            learning_rate=hyperparameter['learning_rate'],
            loss_func=hyperparameter['loss_func'],
            epochs=hyperparameter['epochs'],
            batch_size=hyperparameter['batch_size'],
            initializer=hyperparameter['initializer']
        )

    def get_policy(self, hyperparameter):
        return NN(
            num_hidden_layer=hyperparameter['num_hidden_layer'],
            num_hidden_node=hyperparameter['num_hidden_node'],
            input_shape=hyperparameter['input_shape'],
            activation=hyperparameter['activation'],
            output_size=hyperparameter['output_size'],
            output_activation=hyperparameter['output_activation'],
            optimizer=hyperparameter['optimizer'],
            learning_rate=hyperparameter['learning_rate'],
            loss_func=hyperparameter['loss_func'],
            epochs=hyperparameter['epochs'],
            batch_size=hyperparameter['batch_size'],
            initializer=hyperparameter['initializer']
        )

    def main_loop(self):
        actions = [0, 1]
        # For now the performance metric is not defined so use a for loop instead
        for _ in range(self.rep):
            model_states = []
            model_actions = []
            s = self.env.reset()
            s = np.array(s[0]).reshape(1,4)
            # print(s)
            for _ in range(self.num_it):
                a = np.random.choice(actions, p=self.BC.predict(s)[0])
                result = [0, 0]
                result[a] = 1
                s_next, _, _, _, _ = self.env.step(a)
                s_next = np.array(s_next).reshape(1,4)
                model_states.append(np.concatenate((s, s_next), axis=1))
                model_actions.append(result)
                s = s_next

            input_train_x = np.array([np.array(ele).reshape(-1) for ele in model_states])
            input_train_y = np.array(model_actions)

            print(len(input_train_y))
            print(len(input_train_x))

            self.world_model.train_model(x=input_train_x, y=input_train_y)

            demo_trajectory = []
            demo_actions = []
            s = self.env.reset()
            state = torch.Tensor(s[0])
            state = state.unsqueeze(0)
            not_seen_actions = []
            for _ in range(self.num_it):
                a = self.demoer.get_action(state)
                not_seen_actions.append(a)
                s_next, _, _, _, _ = self.env.step(a)
                demo_trajectory.append([s, s_next])
                s = s_next
            print(not_seen_actions)
            first = False
            for ele in demo_trajectory:
                if not first:
                    ele = [ele[0][0], ele[1]]
                    first = True
                print(ele)
                ele = np.array(ele).reshape(-1).reshape(1,8)
                a = np.random.choice(actions, p=self.world_model.predict(ele)[0])
                result = [0, 0]
                result[a] = 1
                demo_actions.append(result)

            demo_trajectory[0] = [demo_trajectory[0][0][0], demo_trajectory[0][1]]
            policy_input_x = np.array([ele[0] for ele in demo_trajectory])
            policy_input_y = np.array(demo_actions)
            self.BC.train_model(x=policy_input_x, y=policy_input_y)

        self.render()

    def render(self):
        self.env = gym.make('CartPole-v0', render_mode="human")
        s = self.env.reset()

        actions = [0, 1]
        self.env.render()

        s = np.array(s[0]).reshape(1, 4)
        for _ in range(2000):
            a = np.random.choice(actions, p=self.BC.predict(s)[0])
            print(self.BC.predict(s))
            s_next, _, _, _, _ = self.env.step(a)
            self.env.render()
            s_next = np.array(s_next).reshape(1, 4)
            s = s_next

        self.env.close()





