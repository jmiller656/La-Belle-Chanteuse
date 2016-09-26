import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import re
import nltk

input_file = "Swift.txt"
vocab_file = "swift_words.pkl"
tensor_file = "swift_word_data.npy"

with codecs.open(input_file,"r",encoding='utf-8') as f:
	data = f.read()
data = data.replace('\n',' \n ')
punct = ['\n','.',',','"','[',']',':',';','!','?','(',')']
for p in punct:
	data = data.replace(p, ' '+p+' ')
data = re.sub(' +',' ',data).split(' ')
#data = nltk.word_tokenize(data)
counter = collections.Counter(data)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
chars = count_pairs
print chars
chars, _ = zip(*count_pairs)
vocab_size = len(chars)
vocab = dict(zip(chars, range(len(chars))))
with open(vocab_file, 'wb') as f:
	cPickle.dump(chars, f)
tensor = np.array(list(map(vocab.get, data)))
np.save(tensor_file, tensor)
print tensor
