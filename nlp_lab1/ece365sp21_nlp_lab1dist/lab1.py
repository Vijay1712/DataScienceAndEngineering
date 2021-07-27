from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from nltk.corpus import udhr

def get_freqs(corpus, puncts):
    import re
    freqs = {}
    ### BEGIN SOLUTION
    for i in puncts: 
        corpus= corpus.replace(i,' ')
    corpus=re.sub('[0-9]', ' ', corpus)
    corpus=corpus.split() 
    lowerCorpus = []
    for i in corpus:
        lowerCorpus.append(i.lower())

    for i in lowerCorpus:
        if i in freqs:
            freqs[i] +=1
        else:
            freqs[i] = 1
    # freqs=(Counter(lowerCorpus))
    
    ### END SOLUTION
    return freqs

def get_top_10(freqs):
    top_10 = []
    ### BEGIN SOLUTION
    freqs=sorted(freqs.items(), key=lambda item: item[1])
    top10=freqs[-10:]
    for i in range(len(top10)-1, -1, -1):
        # print(top10[i])
        top_10.append(top10[i][0])
    ### END SOLUTION
    return top_10

def get_bottom_10(freqs):
    bottom_10 = []
    ### BEGIN SOLUTION
    freqs=sorted(freqs.items(), key=lambda item: item[1])
    bottom10=freqs[0:10]
    for i in range(len(bottom10)):
        bottom_10.append(bottom10[i][0])
    ### END SOLUTION
    return bottom_10

def get_percentage_singletons(freqs):
    ### BEGIN SOLUTION
    val = 0
    for i in freqs.values():
        if i == 1:
            val+=1
    return (val/len(freqs))*100
    ### END SOLUTION

def get_freqs_stemming(corpus, puncts):
    ### BEGIN SOLUTION
    import re
    freqs = {}
    porter = PorterStemmer()
    ### BEGIN SOLUTION
    for i in puncts: 
        corpus= corpus.replace(i,' ')
    corpus=re.sub('[0-9]', ' ', corpus)
    corpus=corpus.split() 
    lowerStemCorpus = []
    for i in corpus:
        word = i.lower()
        lowerStemCorpus.append(porter.stem(word))

    for i in lowerStemCorpus:
        if i in freqs:
            freqs[i] +=1
        else:
            freqs[i] = 1
    # freqs=(Counter(lowerStemCorpus))
    return freqs
    ### END SOLUTION

def get_freqs_lemmatized(corpus, puncts):
    ### BEGIN SOLUTION
    import re
    freqs = {}
    wordnet_lemmatizer = WordNetLemmatizer()
    ### BEGIN SOLUTION
    for i in puncts: 
        corpus= corpus.replace(i,' ')
    corpus=re.sub('[0-9]', ' ', corpus)
    corpus=corpus.split() 
    lowerLemmaCorpus = []
    for i in corpus:
        word = i.lower()
        lowerLemmaCorpus.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

    for i in lowerLemmaCorpus:
        if i in freqs:
            freqs[i] +=1
        else:
            freqs[i] = 1
    # freqs=(Counter(lowerStemCorpus))
    return freqs
    ### END SOLUTION

def size_of_raw_corpus(freqs):
    ### BEGIN SOLUTION
    return len(freqs)
    ### END SOLUTION

def size_of_stemmed_raw_corpus(freqs_stemming):
    ### BEGIN SOLUTION
    return len(freqs_stemming)
    ### END SOLUTION

def size_of_lemmatized_raw_corpus(freqs_lemmatized):
    ### BEGIN SOLUTION
    return len(freqs_lemmatized)
    ### END SOLUTION

def percentage_of_unseen_vocab(a, b, length_i):
    ### BEGIN SOLUTION
    return len(set(a)-set(b))/length_i

    ### END SOLUTION

def frac_80_perc(freqs):
    ### BEGIN SOLUTION
    totalVal = 0
    for i in freqs.values():
        totalVal+=i
    freqs=sorted(freqs.items(), key=lambda item: item[1], reverse=True)
    tempVal=0
    numberOfKeys=0
    for k, v in freqs:
        tempVal+=v
        numberOfKeys+=1
        if((tempVal/totalVal) >= 0.8):
            break  
    return (numberOfKeys/len(freqs))
    ### END SOLUTION

def plot_zipf(freqs):
    ### BEGIN SOLUTION
    freqs=sorted(freqs.items(), key=lambda item: item[1], reverse=True)
    y=[]
    x=list(range(1,len(freqs)+1))
    for k,v in freqs:
        y.append(v)
    plt.xlabel('rank')
    plt.ylabel('frequency')
    plt.plot(x,y)
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.

def get_TTRs(languages):
    import numpy as np
    import string
    TTRs = {}
    for lang in languages:
        words = udhr.words(lang)
        lowerWords = []
        for i in words:
            lowerWords.append(i.lower())
        tokens = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]
        count = []
        for i in tokens:
            count.append(len(set(lowerWords[:i])))
        TTRs[lang] = count
        ### BEGIN SOLUTION
        ### END SOLUTION
    return TTRs

def plot_TTRs(TTRs):
    ### BEGIN SOLUTION
    import numpy as np
    tokens = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]
    for k in TTRs:
        plt.plot(tokens,TTRs[k],label=k) 
    plt.xlabel('token number')
    plt.ylabel('types')
    plt.xticks(np.arange(100,1400,100))
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.
