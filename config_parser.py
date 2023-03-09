
import json

from function import LayerFunctions
from layer import Layer
from data import LayerData
from typing import Callable, List
from error import MSE
from function import SigmoidFunction
from function import LinearFunction

from network import DeepNeuralNetwork


class NetworkParser:
    def __init__(self, config_file="config/config.json"):
        self.config_file = config_file

    def parse(self) -> List[Layer]:
        with open(self.config_file, 'r') as f:
            nn = json.load(f)
            data = nn["NeuralNetwork"]
            #print(data)

            self.seed = data["seed"]
            self.epochs = data["epochs"]
            self.eval_freq = data["eval_freq"]
            self.batch = data["batch"]
            self.lr = data["lr"]
            self.test_size = data["test_size"]
            self.optimizer = data["optimizer"],
            self.error = data["error"]
            layers: List[Layer] = []
            for layer_data in data["layer"]:
                num_neurons = layer_data["num_neurons"]
                activation = globals()[layer_data["activation"]]()
                layers.append(Layer(LayerData(num_neurons), activation, LayerFunctions(), self.seed))

        return layers

    def create_neural_network(self) -> DeepNeuralNetwork:
        layers = self.parse();
        error = globals()[self.error]()
        #print(error)
        return DeepNeuralNetwork(layers, error, self.seed)








