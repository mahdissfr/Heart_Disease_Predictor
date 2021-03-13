import math

import pandas as pd
import re


class NB:
    def __init__(self):
        self.pos_words = {}
        self.pos_counter = 0
        self.neg_words = {}
        self.neg_counter = 0
        self.stop_words = []
        self.pos_prior_probability = None
        self.neg_prior_probability = None
        f = open("dataset/sw.txt", "r")
        for x in f:
            self.stop_words.append(x.replace("\n", ""))
        f.close()

    def removePunctuation(self, string):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for x in punctuations:
            if x in string:
                string = string.replace(x, "")
        return string

    def removeSW(self, words):
        for word in words:
            if word in self.stop_words:
                words.remove(word)
        return words

    def prepareTheData(self, df):
        for i in range(len(df)):
            label = df.at[i, 'Label']
            text = df.at[i, 'Review'].lower()
            text = self.removePunctuation(text)
            words = text.split(" ")
            words = self.removeSW(words)
            for word in words:
                if label == 'pos':
                    self.pos_counter += 1
                    for word in words:
                        if word not in self.pos_words.keys():
                            self.pos_words[word] = 1
                        else:
                            self.pos_words[word] += 1
                else:
                    self.neg_counter += 1
                    for word in words:
                        if word not in self.neg_words.keys():
                            self.neg_words[word] = 1
                        else:
                            self.neg_words[word] += 1

    def compute_prior_probability(self, alpha):
        self.pos_prior_probability = (self.pos_counter + alpha) / (self.pos_counter + self.neg_counter + 2 * alpha)
        self.neg_prior_probability = (self.neg_counter + alpha) / (self.pos_counter + self.neg_counter + 2 * alpha)

    def compute_conditional_probability(self, word, alpha):
        if word not in self.pos_words:
            self.pos_words[word]=0
        if word not in self.neg_words:
            self.neg_words[word]=0
        cpos = (self.pos_words[word] + alpha) / (self.pos_counter + alpha * len(self.pos_words.keys()))
        cneg = (self.neg_words[word] + alpha) / (self.neg_counter + alpha * len(self.neg_words.keys()))
        return cpos, cneg

    def classify(self, test, alpha):
        self.compute_prior_probability(alpha)
        results = []
        for text in test:
            text = text.lower()
            text = self.removePunctuation(text)
            words = text.split(" ")
            words = self.removeSW(words)
            ppos = 0
            pneg = 0
            for word in words:
                cpos, cneg = self.compute_conditional_probability(word, alpha)
                ppos += math.log2(cpos)
                pneg += math.log2(cneg)
            ppos += math.log2(self.pos_prior_probability)
            pneg += math.log2(self.neg_prior_probability)
            if ppos > pneg:
                results.append('pos')
            else:
                results.append('neg')
        return results


def get_accuracy_score(results, labels):
    cntr = 0
    for i in range(len(results)):
        if results[i]==labels[i]:
            cntr += 1
    return cntr/len(results)


if __name__ == "__main__":
    alpha = 0.12
    dataset = pd.read_csv("dataset/reviews_train.csv")
    nb = NB()
    nb.prepareTheData(dataset)
    test = pd.read_csv("dataset/reviews_test.csv")
    results = nb.classify(test['Review'].to_list(),alpha)
    accuracy = get_accuracy_score(results,test['Label'].to_list())
    print(accuracy)

