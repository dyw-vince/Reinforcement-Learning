from train import *
from test import *
import yaml

if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    runner = Training(config,test=False)
    runner.run()
    tester= Testing(config,test=True)
    tester.run()