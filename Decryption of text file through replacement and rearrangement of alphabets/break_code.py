# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:04:50 2019

@author: Neha
"""

import random
import math
import encode
import sys
import copy
"""
def transProb(corpus):
    tranProbTable = [[0 for i in range(27)] for j in range(27)]
    space = ' '
    temp = ' '
    for ch in corpus:
        if ch == space:
            if temp == space:
                tranProbTable[26][26] += 1
            else:
                tranProbTable[26][ord(temp)-97] += 1
        else:
            if temp == space:
                tranProbTable[ord(ch)-97][26] += 1
            else:
                tranProbTable[ord(ch)-97][ord(temp)-97] += 1
        temp = ch
    for i in range(len(tranProbTable)):

        temp1 = sum(sum(tranProbTable,[]))
        for x in range(len(tranProbTable[0])):
            tranProbTable[i][x] = math.log((tranProbTable[i][x]/temp1) + 1)
    return tranProbTable

def problet(cntletter,corpus):
    cntletter= Counter(corpus)
    characters = [key for key in cntletter]
    total = len(corpus)
    PLi = {}
    for letter in characters:
        PL = cntletter[letter]/total
        PLi[letter] = PL
    return PLi
"""
#Creating probability mtrix of character transitions 
def probability_matrix(corpus):
    chars = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    matrix = [[0 for i in range(len(chars))] for j in range(len(chars))]
    blank = ' '
    curr = ' '
    size = 26
    for c in corpus:
        if c == blank:
            if curr == blank:
                matrix[size][size] += 1
            else:
                matrix[size][ord(curr)-ord('a')] += 1
        else:
            if curr == blank:
                matrix[ord(c)-ord('a')][size] += 1
            else:
                matrix[ord(c)-ord('a')][ord(curr)-ord('a')] += 1
        curr = c
    for i in range(len(matrix)):
        for m in range(len(matrix[0])):
            matrix[i][m] = math.log((matrix[i][m]/sum(sum(matrix,[]))) + 1)
    return matrix
#Function calculates maximum likelihood score
def log_likelihood_score(encoded,matrix):
    s = 0
    blank = ' '
    curr = ' '
    size=26
    for e in encoded:
        if e == blank:
            if curr == blank:
                s += matrix[size][size]
            else:
                s += matrix[size][ord(curr)-ord('a')]
        else:

            if curr == blank:
                s += matrix[ord(e)-ord('a')][size]
            else:
                s += matrix[ord(e)-ord('a')][ord(curr)-ord('a')]
        curr = e
    return s
# Creating replace table 
def create_replacement_table(characters):

    x = characters
    y = characters
    random.shuffle(y)
    rep_table = {}
    for i in range(27):
        rep_table[x[i]] = (y[i])
    return rep_table
#discussed logic with divyanshu agrawal
#Creating Rearrange table
def create_rearrangement_table():
    rea_table = [0,1,2,3]
    random.shuffle(rea_table)
    return rea_table

#Creating rearrange changed table
def rearrange_tab(rearrange_table):
    tab = copy.deepcopy(rearrange_table)
    x,y = random.sample([0,1,2,3],2)
    tab[(x)],tab[(y)]= tab[(y)],tab[(x)]
    return tab
#Creating replace changed table 
def replace_tab(replace_table):
    letters =list(range(ord('a'), ord('z')+1))
    first_alpha,second_alpha = random.sample(letters,2)
    tab = copy.deepcopy(replace_table)
    tab[chr(first_alpha)],tab[chr(second_alpha)]= tab[chr(second_alpha)],tab[chr(first_alpha)]
    return tab
#Break code to find final document
def break_code(encoded,corpus):
    #trans = transProb(corpus)
    #prior = problet(cntletter,corpus)
    chars = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    matrix = probability_matrix(corpus)
    accepted = 0
    final_doc = []
    letters =list(range(ord('a'), ord('z')+1))
    replace_table = dict(zip(map(chr, range(ord('a'), ord('z')+1)), map(chr, letters)))
    rearrange_table = list(range(0,4))
    
    for loop in range(5):
        option= [1,2]
        way = random.choice(option)
        #Initializing tables
        rep_tab = create_replacement_table(chars)
        rea_tab = create_rearrangement_table()
        best_doc = [float('-inf'), ""]
        itera = 0
        while(itera<2000):
            if way == 1:
                new_replace_tab = replace_tab(replace_table)
                new_rearrange_tab= rearrange_table
                
            if way ==2 :
                new_rearrange_tab =rearrange_tab(rearrange_table)
                new_replace_tab= replace_table
            #Calculation of old document likelihood score
            doc_old = encode.encode(encoded,rep_tab,rea_tab)
            p_doc_old = log_likelihood_score(doc_old,matrix)
            #Calculation of new document likelihood score
            doc_new = encode.encode(encoded, new_replace_tab, new_rearrange_tab)
            p_doc_new = log_likelihood_score(doc_new,matrix)
            #Probability check of old doc with new
            if p_doc_new>p_doc_old:
                rep_tab,rea_tab = new_replace_tab,new_rearrange_tab
            #Probability distribution
            if best_doc[0] < p_doc_new:
                best_doc[0],best_doc[1] = p_doc_new,doc_new
            #Incrementing while loop
            itera+=1
            final_doc.append(best_doc)

        final = [float('-inf'), ""]

        for i in range(len(final_doc)):
            if final_doc[i][0] > final[0]:
                final[0],final[1] = final_doc[i][0],final_doc[i][1]

    return final[1]

# Main Function
if __name__== "__main__":

    #if(len(sys.argv) != 4):
    #    raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = encode.read_clean_file(sys.argv[1])
    corpus = encode.read_clean_file(sys.argv[2])
    
    decoded = break_code(encoded, corpus)
    #print(decoded)
    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)
