from collections import Counter

import pandas as pd
from Levenshtein import distance as levDistance


def findMinLevenshteinDistanceSum(solutions):
    distanceSum = {}
    for i, word in enumerate(solutions):
        word = word[0]
        for comparedWord in solutions[i + 1:]:
            comparedWord = comparedWord[0]
            if comparedWord not in distanceSum:
                distanceSum[comparedWord] = 0
            distance = levDistance(comparedWord, word)
            distanceSum[comparedWord] += distance
            distanceSum[word] = distanceSum.get(word, 0) + distance
    sortedDistanceSum = sorted(distanceSum.items(), key=lambda x: x[1])
    with open("MinLevWords.csv", "w", encoding="utf-8") as f1:
        for word, count in sortedDistanceSum[:20]:
            f1.write(word + '\n')


def findMaxSharedLetters(solutions):
    sharedLetterCounts = {}
    for i, word in enumerate(solutions):
        wordStr, wordSet = word
        for comparedWord in solutions:
            comparedStr, comparedSet = comparedWord
            overlap = len(wordSet & comparedSet)
            if comparedStr != wordStr:
                sharedLetterCounts[comparedStr] = sharedLetterCounts.get(comparedStr, 0) + overlap
    sortedSharedLetterCounts = sorted(sharedLetterCounts.items(), key=lambda x: x[1], reverse=True)
    with open("MaxSharedLetters.csv", "w", encoding="utf-8") as f1:
        for word, count in sortedSharedLetterCounts[:20]:
            f1.write(word + '\n')


def findMaxCommonPositions(solutions):
    posMatch = {}
    posMatchCounts = {}
    for i, word in enumerate(solutions):
        wordStr = word[0]
        for comparedWord in solutions:
            comparedStr = comparedWord[0]
            if wordStr == comparedStr:
                continue
            commonPos = sum(w == g for w, g in zip(wordStr, comparedStr))
            if commonPos > 0:
                posMatch[comparedStr] = posMatch.get(comparedStr, 0) + 1
            posMatchCounts[comparedStr] = posMatchCounts.get(comparedStr, 0) + commonPos
    sortedPosMatch = sorted(posMatch.items(), key=lambda x: x[1], reverse=True)
    sortedPosMatchCounts = sorted(posMatchCounts.items(), key=lambda x: x[1], reverse=True)
    with open("MaxMatchWords.csv", "w", encoding="utf-8") as f1, open("MaxMatchLetters.csv", "w", encoding="utf-8") as f2:
        for word, count in sortedPosMatch[:20]:
            f1.write(word + '\n')
        for word, count in sortedPosMatchCounts[:20]:
            f2.write(word + '\n')


def main():
    solutions = [(w, Counter(w)) for w in pd.read_csv("Words.csv", header=None, encoding="utf-8")[0]]
    findMinLevenshteinDistanceSum(solutions)
    findMaxSharedLetters(solutions)
    findMaxCommonPositions(solutions)


if __name__ == "__main__":
    main()
