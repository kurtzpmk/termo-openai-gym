from stable_baselines3.common.monitor import Monitor

from Env import Env
from RLAgent import RLAgent

if __name__ == "__main__":
    env = Env()
    evaluationEnv = Env()
    evaluationEnv = Monitor(evaluationEnv)
    agent = RLAgent(env)
    agent.trainAgent(5000000, evaluationEnv)

    agent.save()

    # env = Env()
    # agent = RLAgent(env, "model/best_model.zip")
    # agent.runTrainedAgent(env, 10)
