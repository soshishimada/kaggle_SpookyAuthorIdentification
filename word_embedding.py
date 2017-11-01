from sklearn.feature_extraction.text import TfidfVectorizer
import csv,codecs
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import stem

#read csv without encoding problem
def csv_unireader(f, encoding="utf-8"):
    for row in csv.reader(codecs.iterencode(codecs.iterdecode(f, encoding), "utf-8")):
        yield [e.decode("utf-8") for e in row]

def read_csv(file,num_line):
  row_vec = []
  labels = []
  ids =[]
  with open(file) as f:
    reader = csv_unireader(f)
    i = 0
    for row in reader:
      if i != 0: #skip the first row
        ids.append(row[0])
        labels.append(row[2])
        row_vec.append(row[1])
      if i >= num_line:
        break
      i += 1
  return ids,labels,np.array(row_vec)

#get ids, labels, sentences from dataset
ids,labels, sentences = read_csv('train.csv',19579)

#store stopwords dataset
stopset = set(stopwords.words('english'))


stemmer = stem.PorterStemmer()
prepared = []
tmp = []
for i in range(len(sentences)):
  #tokenization and anti capitalization
  tokens = word_tokenize(sentences[i].lower())

  #remove stop words and words less than length 1 e.g., "." ":" ";"
  tokens_stoprm = filter(lambda w: len(w) > 1 and w not in stopset,tokens)
  tmp= [stemmer.stem(token) for token in tokens_stoprm]
  prepared.append(" ".join(tmp))
  tmp =[]

#compute tfidf
vect = TfidfVectorizer()
X = vect.fit_transform(prepared)
tfidf = X.toarray()

#concatenate ids, labels, tfidf
last = []
for i in range(len(tfidf)):
  temp = map(str, tfidf[i])
  temp.insert(0,str(labels[i]))
  temp.insert(0,str(ids[i]))
  last.append(temp)
  temp =[]

#write in csv
df = pd.DataFrame(last)
df.to_csv("./embedded_words.csv", index=False)
