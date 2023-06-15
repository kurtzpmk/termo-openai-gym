import warnings
from datetime import datetime as date

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

warnings.filterwarnings("ignore", category=UserWarning)


class RLAgent:
    def __init__(self, env, modelPath=None):
        super().__init__()
        if modelPath is None:
            self.model = PPO("MlpPolicy", env, tensorboard_log="tb/", verbose=1)
        else:
            self.model = PPO.load(modelPath)

    def trainAgent(self, totalSteps, evaluationEnv):
        self.model.learn(totalSteps, tb_log_name="PPO", callback=EvalCallback(evaluationEnv, best_model_save_path="model/", log_path="log/"))

    def runTrainedAgent(self, env, totalEpisodes):
        guessesNum = []
        for episodeCount in range(totalEpisodes):
            observationHistory = env.reset()
            isDone = False
            totalReward = 0
            print(f"Episode: {episodeCount + 1}")
            while not isDone:
                observationHistory, currentReward, isDone, _ = env.step(self.model.predict(observationHistory)[0])
                env.render()
                totalReward += currentReward
            guessesNum.append(env.guessNum)
            print(f"Reward: {totalReward}, Guesses: {env.guessNum}\n")
        print(f"Mean: {np.mean(guessesNum)}, Standard Deviation: {np.std(guessesNum)}")  # Média e Desvio Padrão
