import re
import time
import itertools
import numpy as np
import pandas as pd
import nltk
import vocabulary
from nltk.tokenize import word_tokenize
import unicodedata
from IPython.display import display

def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def pretty_print_matrix(M, rows=None, cols=None, dtype=float):
    display(pd.DataFrame(M, index=rows, columns=cols, dtype=dtype))

def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)
##
# Word processing functions
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "<unk>" # unknown token

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

##
# Data loading functions

def get_sents(name, sname): 
    return [word_tokenize(unicode(s.decode('utf-8').strip())) for s in open('./'+name)],[word_tokenize(s.strip()) for s in open('./'+sname)]
                          
def build_vocab(sents, V=10000):
    words = flatten(sents)
    token_feed = (canonicalize_word(w) for w in words)
    vocab = vocabulary.Vocabulary(False, token_feed, size=V)
    return vocab

def get_train_test_dev_sents(sents, sentis, train=0.5, test=0.25, shuffle=True):
    """Get train, test, and dev sentences."""
    sentences = np.array(sents, dtype=object)
    fmt = (len(sentences), sum(map(len, sentences)))
    print "Loaded %d sentences (%g tokens)" % fmt

    sentiments = np.array(sentis, dtype=object)
    fmt = (len(sentiments), sum(map(len, sentiments)))
    print "Loaded %d sentiments (%g tokens)" % fmt

    if shuffle:
        #rng = np.random.RandomState(shuffle)
        shuf_idx = np.random.permutation(len(sentences))
        sentences = sentences[shuf_idx]
        sentiments = sentiments[shuf_idx]
    split_train_idx = int(train * len(sentences))
    split_test_idx = int((train + test) * len(sentences))
    train_sents = sentences[:split_train_idx]
    test_sents = sentences[split_train_idx:split_test_idx]
    dev_sents = sentences[split_test_idx:]

    fmt = (len(train_sents), sum(map(len, train_sents)))
    print "Training set: %d sentences (%d tokens)" % fmt
    fmt = (len(test_sents), sum(map(len, test_sents)))
    print "Test set: %d sentences (%d tokens)" % fmt
    fmt = (len(dev_sents), sum(map(len, dev_sents)))
    print "dev set: %d sentences (%d tokens)" % fmt

    train_sentis = sentiments[:split_train_idx]
    test_sentis = sentiments[split_train_idx:split_test_idx]
    dev_sentis = sentiments[split_test_idx:]

    fmt = (len(train_sentis), sum(map(len, train_sentis)))
    print "Training set: %d sentiments (%d tokens)" % fmt
    fmt = (len(test_sentis), sum(map(len, test_sentis)))
    print "Test set: %d sentiments (%d tokens)" % fmt
    fmt = (len(dev_sentis), sum(map(len, dev_sentis)))
    print "dev set: %d sentiments (%d tokens)" % fmt
    
    return train_sents, test_sents, dev_sents, train_sentis, test_sentis, dev_sentis

def preprocess_sent_sentis(sents, sentis, vocab, svocab):
    # Add sentence boundaries, canonicalize, and handle unknowns
    words = ["<s>"] + flatten(s + ["<s>"] for s in sents)
    words = [canonicalize_word(w, wordset=vocab.word_to_id)
             for w in words]
    
    idx = 0
    swords = [sentis[idx][0]]  #ref to first <s> at the very beginning
    for s in sents:
        next_sword = sentis[idx][0]
        for w in s:
            swords.append(next_sword) #ref to all words in a sent
        swords.append(next_sword)     #ref to the end of each sent
        idx = idx + 1
                          
    return np.array(vocab.words_to_ids(words)), np.array(svocab.words_to_ids(swords))

##
# Use this function
def load_data(name, sname, train=0.5, test=0.25, V=10000, Z=8, shuffle=True):
    """Load a named corpus and split train/test along sentences."""
    sents, sentis = get_sents(name, sname)   
    vocab = build_vocab(sents, V)
    svocab = vocabulary.Vocabulary(True)
    
    train_sents, test_sents, dev_sents, train_sentis, test_sentis, dev_sentis = get_train_test_dev_sents(sents, sentis, train, test, shuffle)
    train_ids, train_sids = preprocess_sent_sentis(train_sents, train_sentis, vocab, svocab)
    test_ids, test_sids = preprocess_sent_sentis(test_sents, test_sentis, vocab, svocab)
    dev_ids, dev_sids = preprocess_sent_sentis(dev_sents, dev_sentis, vocab, svocab)

    return vocab, svocab, train_ids, train_sids, test_ids, test_sids, dev_ids, dev_sids, train_sents, train_sentis, test_sents, test_sentis, dev_sents, dev_sentis

##
# Use this function
def batch_generator(ids, sids, batch_size, max_time):
    """Convert ids to data-matrix form."""
    # Clip to multiple of max_time for convenience
    clip_len = ((len(ids)-1) / batch_size) * batch_size
    input_w = ids[:clip_len]     # current word
    target_y = sids[:clip_len]  # the sents
    print 'in batch_generator', len(ids), len(sids), len(input_w), len(target_y), clip_len, batch_size
    # Reshape so we can select columns
    input_w = input_w.reshape([batch_size,-1])
    target_y = target_y.reshape([batch_size,-1])

    # Yield batches
    for i in xrange(0, input_w.shape[1], max_time):
	yield input_w[:,i:i+max_time], target_y[:,i:i+max_time]


