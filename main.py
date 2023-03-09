from config_parser import NetworkParser
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


from trainer import Trainer


def main():
    '''
    Predict the price based on California real estate data
    :return:
    '''

    parser = NetworkParser()
    neural = parser.create_neural_network()
    #print(neural)

    ca_housing = fetch_california_housing()
    input = ca_housing.data
    target = ca_housing.target
    # f

    s = StandardScaler()
    input = s.fit_transform(input)
    trainer = Trainer(neural, parser)
    trainer.fit(input, target)

if __name__ == '__main__':
    main()