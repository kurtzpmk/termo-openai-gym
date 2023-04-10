from Env import Env
from RLAgent import RLAgent

steps = []
correct = 0
if __name__ == "__main__":
    for i in range(100):
        env = Env()
        agent = RLAgent(env, modelPath="model/best_model.zip")
