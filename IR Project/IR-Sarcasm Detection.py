import nltk
import json
import math

def loadData():
    for line in open('Sarcasm_Headlines_Dataset.json','r'):
        yield eval(line)

Data=list(loadData())

print(Data[0])
print(Data[0]['headline'])
print(Data[0]['is_sarcastic'])    

X=[]
Y=[]

for i in range(len(Data)):
    X.append(Data[i]['headline'])
    Y.append(Data[i]['is_sarcastic'])    

print(X[2])
print(Y[2])
size=len(X)
size

#Tokens
wordsTORemove=[]
tokens=[]
for i in range(len(X)):
    temp=nltk.word_tokenize(X[i])
    for j in range(len(temp)):
        tokens.append(temp[j])

print(len(tokens))

#Decapitalized
tokens=[element.lower() for element in tokens]

#Remove Special characters
removetable=str.maketrans("", "", "'!@#$%^&*()_=-\|][:';:,<.>/?`~0123456789")
tokens=[x.translate(removetable) for x in tokens]

#Importing Stoplist
stopWord=open("Stopword-List.txt").read()
stopWord=nltk.word_tokenize(stopWord)
# Removing StopWords
tokens=[x for x in tokens if x.isalnum() and x not in stopWord]

#word Frequency
fdist=nltk.FreqDist(tokens)

# Unique Tokens 
tokens=list(set(tokens))
print(len(tokens))

#Counting Frequency
#Removing words freq <=3
for words in tokens:
    if(fdist[words]<=3):
        tokens.remove(words)
        wordsTORemove.append(words)

len(tokens)

#Removing Token Tags
tagg=nltk.tag.pos_tag(tokens)
taggs=set()
for word,type1 in tagg:
    if(type1=='NNP' or type1=='FW' or type1=='PRP'):
        taggs.add(word)    

tokens=list(set(tokens)-taggs)
size=len(tokens)
size

#Removing words with length 2
for words in tokens:
    if len(words)<=2:
        tokens.remove(words)
        wordsTORemove.append(words)

len(tokens)

#Tokens
docToken=[]
for i in range(size):
    docToken.append(nltk.word_tokenize(X[i]))    

print(docToken[106])

#Decaptilized Doc Wise
for x in range(size):
    docToken[x]=[element.lower() for element in docToken[x]]

#Remove Special characters Doc Wise
removetable=str.maketrans("", "", "'!@#$%^&*()_=-\|][:';:,<.>/?`~")
for x in range(size):
    docToken[x]=[y.translate(removetable) for y in docToken[x]]

#removing StopWords
for x in range(size):
    docToken[x]=[y for y in docToken[x] if y.isalnum() and y not in stopWord]

#Removing Token Tags
for x in range(size):
    docToken[x]=list(set(docToken[x])-taggs)

# Counting Frequency
# Removing words freq <=3
for x in range(size):
    for words in docToken[x]:
        if len(words)<=2:
            docToken[x].remove(words)
        elif words in wordsTORemove:
            docToken[x].remove(words)

print(docToken[106])


docV={}
for x in range(size):
    docV[x]=dict.fromkeys(tokens,0) 
    
#Term Frequency in  a document
for x in range(size):
    for word in docToken[x]:
        try:
            docV[x][word]+=1
        except KeyError:
            pass

print(docV[106]['grants'])

#tf
tfDocV={}
for x in range(size):
    tfDocV[x]={}
    for word,count in docV[x].items():
        tfDocV[x][word]=count
        

print(len(tfDocV[1]))

#unique Token Doc wise
for x in range(size):
    docToken[x]=set(docToken[x])
    docToken[x]=list(set(docToken[x]))

wordDcount=dict.fromkeys(tokens,0)
for word in tokens:
    for x in range(size):
        if word in docToken[x]:
            wordDcount[word]+=1

len(wordDcount)

#idf            
idfDict = {}
for word in tokens:
    if wordDcount[word]>0:
        count=wordDcount[word]
        if count>size:
            count=size
    
    idfDict[word]=math.log(size/count)        

len(idfDict)

#tf-idf    
tfidf={}
for x in range(size):
    tfidf[x]={}
    for word in docV[x]:
        tfidf[x][word]=tfDocV[x][word]*idfDict[word]

print(tfidf[106]['grants'])


bigram=[]
for i in range(size):
    ls=list(nltk.bigrams(docToken[i]))
    for j in ls:
        bigram.append(list(j))

len(bigram)


















