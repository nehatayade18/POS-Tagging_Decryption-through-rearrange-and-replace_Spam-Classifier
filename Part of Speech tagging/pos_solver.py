###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids: Viral Prajapati vkprajap
#
# (Based on skeleton code by D. Crandall)
#

import random
import math
import copy
import numpy as np
from collections import Counter

class Solver:
    
# calculating the priors for all the POS tags or labels
    def probS(self, cntTags, tags):
        posTags = [key for key in cntTags]
        total = len(tags)
        PSi = {}
        for tag in posTags:
            Ps = cntTags[tag] / total
            PSi[tag] = Ps
        return PSi

#calculating initial probability of all the first word of sentences in the data for viterbi approach
    def initProb(self, data):
        temp = []
        i = 0
        for i in range(len(data)):
            temp.append(data[i][1][0])
        cntTemp = Counter(temp)
        initProbs = []
        for k, v in cntTemp.items():
            initProbs.append((v/len(temp), k))
        return initProbs

#calculating transition probability of all the labels from the data
    def transProb(self, tags, cntTags):
        posTags = [key for key in cntTags]
        tranProbTable = []
        for i in posTags:
            temp1 = []
            for j in range(len(posTags)):
                count = 0.00000001
                for k in range(len(tags)-1):
                    if tags[k] == i and tags[k+1] == posTags[j]:
                        count+=1
                temp1.append(((count/cntTags[i]),posTags[j]))
            tranProbTable.append(temp1)
        return tranProbTable

#calculating emission probability of the words in the test data or given sentence
    def emissProb(self, sentence, words, tags, cntTags):
        posTags = [key for key in cntTags]
        emissProbTable = []        
        for i in range(len(posTags)):
            temp1 = []
            for j in range(len(sentence)):
                if i == 1:
                    temp1.append(0.0001)
                else:
                    temp1.append(0.00000001)
                #temp1.append(0.01)
            emissProbTable.append(temp1)
            
        for x in range(len(sentence)):
            for y in range(len(words)):
                if sentence[x] == words[y]:
                    for z in range(len(posTags)):
                        if tags[y] == posTags[z]:
                            ind = z
                            break
                    emissProbTable[ind][x] = emissProbTable[ind][x] + 1
        
        for k in range(len(cntTags)):
            for l in range(len(emissProbTable[0])):
                emissProbTable[k][l] = float(emissProbTable[k][l] / cntTags[posTags[k]])
    
        return emissProbTable

# Predicting tags using simplified bayes net
    def simpleModel(self, sentence, words, tags, cntTags, PSi, emissProbTable):
        tagsList = []
        prob = []
        tagList = [key for key in cntTags]
        for i in range(len(sentence)):
            probList = []
            for j in range(len(tagList)):
                temp = emissProbTable[j][i] * PSi[tagList[j]]
                probList.append(temp)
            tagsList.append(tagList[probList.index(max(probList))])
            prob.append(max(probList))
        
        try:
            temp1 = math.log(sum(prob))
        except:
            temp1 = 1
        
        return tagsList, temp1
    
# Predicting tags using Viterbi approach by calculating MA
#This Viterbi Algorithm has been referenced from https://en.wikipedia.org/wiki/Viterbi_algorithm        
    def viterbiModel(self, sentence, words, tags, cntTags, initProbs, tranProbTable, emissProbTable):
        prob = []
        tagList = [key for key in cntTags]
        predictions = [{}]
        for i in range(len(tagList)):
            predictions[0][i] = {"prob": initProbs[i][0] * emissProbTable[i][0], "prev": None}
        for j in range(1, len(sentence)):
            predictions.append({})
            for k in range(len(tagList)):
                maxTranProb = predictions[j-1][0]["prob"]*tranProbTable[k][0][0]
                prevState = 0
                for l in range(len(tagList[1:])):
                    tranProb = predictions[j-1][l]["prob"]*tranProbTable[l][k][0]
                    if tranProb > maxTranProb:
                        maxTranProb = tranProb
                        prevState = l

                max_prob = maxTranProb * emissProbTable[k][j]
                predictions[j][k] = {"prob": max_prob, "prev": prevState}
                
        temp = []
        maxProb = max(value["prob"] for value in predictions[-1].values())
        previous = None
        for state, proba in predictions[-1].items():
            if proba["prob"] == maxProb:
                temp.append(state)
                previous = state
                break
        for x in range(len(predictions)-2, -1, -1):
            temp.insert(0, predictions[x + 1][previous]["prev"])
            prob.insert(0, predictions[x + 1][previous]["prob"])
            previous = predictions[x + 1][previous]["prev"]

        predResult = []
        for y in temp:
            predResult.append(tagList[y])
            
        try:
            temp1 = math.log(maxProb)
        except:
            temp1 = 1
        
        return predResult, temp1

# Predicting tags using Gibbs sampling approach
    def gibbsModel(self, sentence, words, tags, cntTags, tranProbTable, PSi, emissProbTable):
        tagList = [key for key in cntTags]
        result = ['noun'] * len(sentence)
        resultTable = []
        for n in range(1000):
            finalProb = 0
            for i in range(len(sentence)):
                temp1 = []
                if i == 0:
                    for j in range(len(tagList)):
                        probability = emissProbTable[j][i] * PSi[tagList[j]]
                        temp1.append(probability)
                elif i == len(sentence) - 1:
                    for j in range(len(tagList)):
                        k = tagList.index(result[i-1])
                        l = tagList.index(result[0])
                        probability = emissProbTable[j][i]*PSi[tagList[j]]*tranProbTable[k][j][0]*tranProbTable[l][j][0]
                        temp1.append(probability)
                else:
                    for j in range(len(tagList)):
                        k = tagList.index(result[i-1])
                        probability = emissProbTable[j][i]*PSi[tagList[j]]*tranProbTable[k][j][0]
                        temp1.append(probability)
                rand = random.uniform(0,1)
                temp2 = 0
                ind = temp1.index(max(temp1))
                for z in range(len(temp1)):
                    temp1[z] = temp1[z] / sum(temp1)
                    temp2 += temp1[z]
                    if rand <= temp2:
                        ind = z
                        break
                result[i] = tagList[ind]
                if n == 999:
                    try:
                        prob = math.log(probability)
                    except:
                        prob = 1
                    finalProb += prob
            
            temp = copy.copy(result)
            resultTable.append(temp)

        resultTable = np.array(resultTable)
        resultTableT = resultTable.T
        predTags = []
        for i in range(len(resultTableT)):
            TagCount = Counter(resultTableT[i])
            predTags.append(TagCount.most_common(1)[0][0])

        return predTags, finalProb/3.2

    def posterior(self, model, sentence, label, probS, probH, probC):
        if model == "Simple":
            return probS
        elif model == "Complex":
            return probC
        elif model == "HMM":
            return probH
        else:
            print("Unknown algo!")

    
    def train(self, data):
        tags = []
        words = []
        for i in data:
            for j in i[1]:
                tags.append(j)
            for k in i[0]:
                words.append(k)
        cntTags = Counter(tags)
        initProbs = self.initProb(data)
        tranProbTable = self.transProb(tags, cntTags)
        PSi = self.probS(cntTags, tags)
        return words, tags, cntTags, initProbs, tranProbTable, PSi

    def simplified(self, sentence, words, tags, cntTags, PSi, emissProbTable):
        results, prob = self.simpleModel(sentence, words, tags, cntTags, PSi, emissProbTable)
        return results, prob

    def complex_mcmc(self, sentence, words, tags, cntTags, tranProbTable, PSi,  emissProbTable):
        results, prob = self.gibbsModel(sentence, words, tags, cntTags, tranProbTable, PSi,  emissProbTable)
        return results, prob
        
    def hmm_viterbi(self, sentence, words, tags, cntTags, initProbs, tranProbTable, emissProbTable):
        results, prob = self.viterbiModel(sentence, words, tags, cntTags, initProbs, tranProbTable, emissProbTable)
        return results, prob


    def solve(self, model, sentence, words, tags, cntTags, initProbs, tranProbTable, PSi, emissProbTable):
        if model == "Simple":
            return self.simplified(sentence, words, tags, cntTags, PSi, emissProbTable)
        elif model == "Complex":
            return self.complex_mcmc(sentence, words, tags, cntTags, tranProbTable, PSi, emissProbTable)
        elif model == "HMM":
            return self.hmm_viterbi(sentence, words, tags, cntTags, initProbs, tranProbTable, emissProbTable)
        else:
            print("Unknown algo!")
