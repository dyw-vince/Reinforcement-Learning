from train import *
from test import *
import yaml
os.environ["http_proxy"] = "http://10.54.14.112:7890"
os.environ["https_proxy"] = "http://10.54.14.112:7890"
if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # runner = Training(config,test=False)
    # runner.run()
    tester= Testing(config,test=True)
    tester.run()