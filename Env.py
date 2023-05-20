import random
from collections import Counter

import gym
import numpy as np
import unicodedata
from colorama import Fore
from gym import spaces
from gym.spaces import MultiDiscrete


def readWords(path):
    with open(path, "r", encoding="utf-8") as file:
        words = file.read().splitlines()
    return words


def readWordsAndLettersFrequency(path):
    words = []
    with open(path, "r", encoding="utf-8") as file:
        for w in file.read().splitlines():
            words.append((w, Counter(w)))
    return words


def equalsIgnoreAccent(stLetter, ndLetter):
    if stLetter.lower() == "c" and ndLetter.lower() == "ç" or stLetter.lower() == "ç" and ndLetter.lower() == "c":
        return True
    return unicodedata.normalize("NFD", stLetter)[0].lower() == unicodedata.normalize("NFD", ndLetter)[0].lower()


class Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.numOfAttempts = 6
        self.episodeCount = 0
        self.choices = []
        self.guessNum = 0
        self.guessList = readWordsAndLettersFrequency("word/Words.csv")
        self.wordList = readWords("word/Words.csv")
        self.action_space = MultiDiscrete((5, 20))
        self.observation_space = spaces.Box(0, len(self.guessList), shape=(self.numOfAttempts + 3,), dtype=np.int32)
        self.maxMatchWords = readWords("word/MaxMatchWords.csv")
        self.maxMatchLetters = readWords("word/MaxMatchLetters.csv")
        self.maxSharedLetters = readWords("word/MaxSharedLetters.csv")
        # Reset
        self.observationHistory = None
        self.solution = None
        self.guesses = None
        self.wrongPositions = None
        self.correctLetters = None
        self.incorrectLetters = None

    def reset(self):
        self.solution = self.wordList[np.random.randint(len(self.wordList))]
        self.guessNum = 0
        self.guesses = []
        # 0: len(self.correctLetters)
        # 1: (sum([len(v) for v in self.wrongPositions.values()]))
        # 2: (len(self.incorrectLetters))
        # ...: Indices das Palavras das Tentativas Anteriores
        self.observationHistory = [0, 0, 0] + [-1] * self.numOfAttempts
        self.correctLetters = {}
        self.wrongPositions = {}
        self.incorrectLetters = []
        self.episodeCount += 1
        if self.episodeCount % 100 == 0: print(f"\nEpisode Count: {self.episodeCount}")  # Debug
        return self.observationHistory

    def processGuess(self, guess):
        isDone = False
        wrongPositionsAddedCurrentStep = []
        if all(equalsIgnoreAccent(a, b) for a, b in zip(guess, self.solution)):
            isDone = True
            currentReward = self.numOfAttempts
            self.observationHistory = [5, 0, 0] + self.observationHistory[3:]
            self.observationHistory[self.guessNum] = self.wordList.index(guess)
            self.correctLetters = dict(zip(range(5), self.solution))
        elif guess in self.guesses:
            currentReward = (self.numOfAttempts / 3) * -1
        else:
            guessChars = list(guess)
            solutionChars = list(self.solution)
            for i in range(5):
                if equalsIgnoreAccent(guessChars[i], solutionChars[i]):
                    self.correctLetters[i] = guessChars[i]
                elif any(equalsIgnoreAccent(guessChars[i], s) for s in solutionChars):
                    if guessChars[i] in wrongPositionsAddedCurrentStep:
                        continue
                    indexes = [j for j, x in enumerate(solutionChars) if equalsIgnoreAccent(x, guessChars[i])]
                    accountedFor = True
                    for index in indexes:
                        if not equalsIgnoreAccent(solutionChars[index], guessChars[index]):
                            accountedFor = False
                    if not accountedFor:
                        if guessChars[i] in self.wrongPositions:
                            self.wrongPositions[guessChars[i]].append(i)
                        else:
                            self.wrongPositions[guessChars[i]] = [i]
                        if len(indexes) == len(self.wrongPositions[guessChars[i]]):
                            wrongPositionsAddedCurrentStep.append(guessChars[i])
                else:
                    self.incorrectLetters.append(guessChars[i])
            self.observationHistory = [len(self.correctLetters), sum([len(v) for v in self.wrongPositions.values()]),
                                       len(self.incorrectLetters)] + self.observationHistory[3:]
            self.observationHistory[3 + self.guessNum] = self.wordList.index(guess)
            currentReward = -1
        self.guessNum += 1
        self.guesses.append(guess)
        if self.guessNum == self.numOfAttempts: isDone = True
        if self.episodeCount % 100 == 0: self.render()  # Debug
        return self.observationHistory, currentReward, isDone, {}

    def step(self, action):
        action, choice = action
        actionGuessMap = {
            0: self.guessList[choice][0],  # Inicial
            1: self.maxMatchWords[choice],  # Maior Correspondência de Palavras
            2: self.maxMatchLetters[choice],  # Maior Correspondência de Letras
            3: self.maxSharedLetters[choice],  # Maior Compartilhamento de Letras
            4: random.choice(self.findCandidateWords())[0],  # Aleatório
        }
        return self.processGuess(actionGuessMap.get(action))

    def isCandidate(self, wordChars):
        correctPositions = self.correctLetters.keys()
        for index, letter in enumerate(wordChars):
            if index in correctPositions and not equalsIgnoreAccent(self.correctLetters[index], letter):
                return False
            elif any(equalsIgnoreAccent(letter, incorrect_letter) for incorrect_letter in self.incorrectLetters):
                return False
        for correctLetter, incorrectPositions in self.wrongPositions.items():
            letterIndexes = [j for j, x in enumerate(wordChars) if equalsIgnoreAccent(x, correctLetter)]
            if not any(equalsIgnoreAccent(correctLetter, x) for x in wordChars) or np.all(
                    letterIndexes == incorrectPositions):
                return False
        return True

    def findCandidateWords(self):
        return [(word, freq) for word, freq in self.guessList if
                word not in self.guesses and self.isCandidate(list(word))]

    def printResult(self, letter: str, index):
        if index in self.correctLetters and self.correctLetters[index] == letter:
            print(Fore.GREEN + f"{letter.upper()}", end="")
        elif letter in self.wrongPositions and index in self.wrongPositions[letter]:
            print(Fore.YELLOW + f"{letter.upper()}", end="")
        else:
            print(Fore.WHITE + letter.upper(), end="")

    def render(self, mode="human"):
        lastGuess = self.guesses[-1]
        [self.printResult(letter, index) for index, letter in enumerate(lastGuess)]
        print(Fore.RESET)

    def close(self):
        pass
