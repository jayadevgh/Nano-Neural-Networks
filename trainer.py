from copy import deepcopy

import sklearn.utils
from sklearn.model_selection import train_test_split

from network import DeepNeuralNetwork
from config_parser import NetworkParser
from optimizer import SGD
from error import MSE


class Trainer:
    def __init__(self, network: DeepNeuralNetwork, parser : NetworkParser):
        self.network = network
        self.parser = parser
        self.last_err = 1e9

    def fit(self, input, target) -> None:
        '''Split the data based on the config test size parameter as training and testing data after shuffling it'''
        #input, target = input[:64, :], target[:64]
        input_train, input_test, target_train, target_test =\
            train_test_split(input, target, test_size=self.parser.test_size, random_state=68987)
        target_train, target_test = target_train.reshape(-1, 1), target_test.reshape(-1, 1)
        for iter in range(self.parser.epochs):
            prev_model = deepcopy(self.network)
            '''Train the batches and update the weights '''
            batches = sklearn.utils.gen_batches(input_train.shape[0], self.parser.batch)
            for idx in batches:
                input_batch, target_batch = input_train[idx], target_train[idx]
                #print("Epoch batch", iter, idx)
                self.network.train(input_batch, target_batch)

               # optim = globals()[self.parser.optimizer]()
                self.network.optimize(SGD())

            if (iter + 1) % self.parser.eval_freq == 0:
                #print("input_test", input_test)
                test_preds = self.network.feed_forward(input_test)

                err = self.network.error.feed_forward(test_preds, target_test)

                if err < self.last_err:
                    print(f"Validation error:{err:.3f}", "number of epochs", iter + 1)
                    self.last_err = err
                else:
                    print(f"Validation error:", err, "early stop at number of epochs", iter + 1)
                    self.net = prev_model
                    break










