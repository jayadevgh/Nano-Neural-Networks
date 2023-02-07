
import json
from layer import Layer, DeepNetworkLayer, NetworkLayer
from typing import Callable, List

class NeuralNetworkParser:
    def __init__(self, config_file="config/config.json"):
        self.config_file = config_file


    def parse(self) -> List[Layer]:
        with open(self.config_file, 'r') as f:
            data = json.load(f)
            self.seed = data["seed"]
            self.epochs = data["epochs"]
            self.eval = data["eval"]
            self.batch = data["batch"]
            layers: List[Layer] = None
            for layer_data in data["layer"]:
                num_neurons = layer_data["num_neurons"]
                activation = layer_data["activation"]
                layers.append(NetworkLayer(num_neurons, activation))

        return layers




