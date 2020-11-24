# Author:
# Lakshmi Prasanna Gundabolu
# Date: 03/07/2020
#importing the required libraries
import sys
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from random import random
import numpy as np
import pandas as pd
import re
#printing information regarding our program
print("This program generates random sentences based on an Ngram model.")
for p in range(len(sys.argv[1:])):
    if(p==0):
        print("Number of Ngrams chosen= "+sys.argv[1])
        ngrams = int(sys.argv[1])
    elif(p==1):
        print("Number of sentences to be displayed= "+sys.argv[2])
        sentences = sys.argv[2]
    else:
        print("Text File number :"+ str(p-1)+"= "+ sys.argv[p+1])

#reading all the text files given as an input into input_data
input_data=""
for y in range(2,len(sys.argv[1:])):
    with open(sys.argv[y+1],encoding="latin-1") as f_open:
        input_data= input_data + (f_open.read().lower())
#using regular expressions to substitute the repeated data
text= re.sub(r'(-|\'|_|\.|\/|\*)', r' \1 ',input_data)
#creating a list
tokens=[]
#tokenizing the words
tokens = (word_tokenize(text))
#frequency distribution of tokens
fdist = FreqDist(tokens)
#creating a end list with end tokens
end = ['.','!','?']
#creating a start list tp store start tokens
start = []
#creating logic for unigram
if(ngrams==1):
    count =0
    #finding start tokens that come after end tokens
    for word in tokens:
        if(word=="." or word=="!" or word=="?"):
            if(count== len(tokens)-1):
                break
            #appending the tokens to start list
            start.append(tokens[count+1])
        count+=1
    start = set(start) #to convert start list into iterable form
    #remove end tokens if any from start list
    templist=[]
    for word in start:
        if word in end:
            continue
        templist.append(word)
    start = templist
    #adding start to freqdist dictionary
    count = 0
    for key,value in fdist.items():
        for word in start:
            if(key==word):
                count= count+ value
    fdist["<start>"]= count
    #adding end to freqdist dictionary
    count = 0
    for key,value in fdist.items():
        for word in end:
            if(key==word):
                count= count+ value
    fdist["<end>"]= count
    #removing start and end tokens from fdist dictionary
    fdist2 = {}  #for removing unnecessary tokens which are not words.
    for key,value in fdist.items():
        if key in end: continue #Key might be ., !, ? for removing end tokens
        elif key in start: continue #key might be any regular expression for removing start tokens
        fdist2[key]= value

    #creating probability distribution for finding probabilities of every word in corpus

    probdist={}
    for key,value in fdist2.items(): #fdist2 contains only words without stop and start tokens
        probdist[key]=value/len(tokens) 
    #probdist  
    #create interval prob dist
    intervalprobdist={}
    sum=0
    for key,value in probdist.items():
        sum=sum+value
        intervalprobdist[key]=sum

    #Unigram model
    for r in range(int(sentences)):
        start_c=1
        end_c=0
        line="Sentence "+str(r+1)+" :"
        while(end_c==0):
            rand1 = random()
            if(start_c==1):
                #picking random sentences from start list
                line= line+" "+np.random.choice(list(start))
                start_c=0
            for key,value in intervalprobdist.items():
                if(rand1<value):
                    if(key=="<start>"):
                        key = np.random.choice(list(start))
                    elif(key=="<end>"):
                        key = np.random.choice(list(end))
                        end_c=1
                    line=line+" "+key
                    break
        #printing the random sentences for unigram model     
        print(line)    

else:
     #creating n gram model
     #computing n-1 tokens
    tokens_n1=[]
    #for each position across the range of length of token list 
    for pos in range(len(tokens)):
        if(pos==len(tokens)-(ngrams-2)):
            break
        line = ""
        for pos2 in range(ngrams-1):
            if(pos2==0):
                line = str(tokens[pos+pos2])
            else:
                line = line + " " + str(tokens[pos+pos2])
        tokens_n1.append(line)
    #to promopt to console that n-1 table is created
    print("n-1 table has been created")

    # computing n-1 token frequency table
    fdist_n1 = FreqDist(tokens_n1)
    #create n gram tokens
    if(ngrams>1):
        tokens_n=[]
        #reusing the above logic for creating n tables
        for pos in range(len(tokens)):
            if(pos==len(tokens)-(ngrams-1)):
                break
            line = ""
            for pos2 in range(ngrams):
                if(pos2==0):
                    line = str(tokens[pos+pos2])
                else:
                    line = line + " " + str(tokens[pos+pos2])
            tokens_n.append(line)
        #to promopt to console that n table is created
        print("n table created")
    #computing n token frequency table
    fdist_n = FreqDist(tokens_n)
    #create dataframe for calculating next word probability
    d = {}
    #using pandas data frame
    df = pd.DataFrame(data=d, index=set(tokens_n1),columns=set(tokens))
    #converting data frame to 2d list to fasten processing time
    array = df.values.tolist()
    #create n gram data frame indicating probabilities with laplace smoothing
    prob = 0
    #creating a list to store the sum of rows
    row_sum=[]
    #creating a variable to store the length of takens
    len_tokens = len(set(tokens))
    #creating a variable to store the current row
    row_n=0
    for rows in df.index.values:
        sum=0
        col_n=0
        #for iterating columns in data frame
        for cols in df.columns.values:
            line = str(rows)+" "+str(cols)
            value = fdist_n[line]
            #calculating the probabilities of values in dictionary
            prob=(value+1)/(fdist_n1[rows]+len_tokens)
            array[row_n][col_n]=prob
            sum=sum+prob
            col_n +=1
        row_n += 1
        #appending sum values to list
        row_sum.append(sum)
    #prompting to the command line that laplace table is created
    print("Laplace probability table created")
    #calculate absolute probabilities
    for row in range(len(df.index.values)):
        prob=0
        for col in range(len(df.columns.values)):
            array[row][col]=array[row][col]/row_sum[row]
            prob = prob+array[row][col]
            array[row][col] = prob
    #converting the list to pandas data frame
    df = pd.DataFrame(data=array, index=set(tokens_n1),columns=set(tokens))
    #creating start and end lists
    end = ['.','!','?']
    start = []
    #for each position across the range of length of tokens list
    for pos in range(len(tokens)):
        if tokens[pos] in end:
            if(pos>=len(tokens)-(ngrams-1)):
                break
            line = ""
            for pos2 in range(ngrams-1):
                if(pos2==0):
                    line = str(tokens[pos+pos2+1])
                else:
                    line = line + " " + str(tokens[pos+pos2+1])
            start.append(line)
    start = set(start)
    #print sentences
    print("printing the sentences")
    for x in range(int(sentences)):
        line =np.random.choice(list(start))
        line2 =line
        end_c=0
        while(end_c==0):
            row_n=0
            for rows in df.index.values:
                if(rows==line2):
                    col_n=0
                    #creating a random variable
                    rand = random()
                    for cols in df.columns.values:
                        if(rand<array[row_n][col_n]):
                            prev_line = line
                            prev_line2 = line2
                            line = line + " " + cols
                            re_line = rows + " " + cols
                            # spotting last n-1 words
                            res = re.match(r'^\S+(.*)',re_line)
                            line2 = res[1].strip()                           
                            break
                        col_n +=1
                    if line2 not in tokens_n1:
                        line = prev_line
                        line2= prev_line2 
                        break
                    elif re.search(r'(\.|\?|!)',line):
                        end_c=1
                        break
                row_n+=1
        #printing the sentences        
        print("Sentence "+str(x+1)+": "+line)