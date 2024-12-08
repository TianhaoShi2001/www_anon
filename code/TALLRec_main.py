from config import config
from utils import *
from datasets import Datasets_builder
from base_models import CoLLM
from model_runner import Runner

def generate_random_numbers(length):
    random_numbers = [random.random() for _ in range(length)]
    return random_numbers



def main():
    print('config', config)

    init_distributed_mode(config)
    setup_seeds(config)
    setup_logger()

    data_builder = Datasets_builder(config)
    datasets = data_builder.datasets

    model = CoLLM(config)
    model_runner = Runner(config, model = model, datasets = datasets)
    model_runner.fit()

if __name__ == "__main__":
    main()
