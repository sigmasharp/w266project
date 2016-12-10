import re
import time
import itertools
import numpy as np
import pandas as pd
from IPython.display import display
senti = [0, 1, 3, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 61, 62, 63, 70, 71, 72, 75, 76, 78]
sentid = {0: 0, 1: 1, 3: 2, 5: 3, 9: 4, 10: 5, 11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16, 22: 17, 23: 18, 24: 19, 25: 20, 26: 21, 27: 22, 28: 23, 29: 24, 30: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 45: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 59: 53, 61: 54, 62: 55, 63: 56, 70: 57, 71: 58, 72: 59, 75: 60, 76: 61, 78: 62}

def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))

def pretty_print_matrix(M, rows=None, cols=None, dtype=float):
    """Pretty-print a matrix using Pandas.

    Args:
      M : 2D numpy array
      rows : list of row labels
      cols : list of column labels
      dtype : data type (float or int)
    """
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
import nltk
import vocabulary

def get_corpus(name="brown"):
    if name=='brown':
        return nltk.corpus.__getattr__(name)
    return nltk.corpus.PlaintextCorpusReader('./', name)

def build_vocab(corpus, V=10000):
    token_feed = (canonicalize_word(w) for w in corpus.words())
    vocab = vocabulary.Vocabulary(token_feed, size=V)
    return vocab

def build_vocab_senti(sentis, Z=63):
    vocab = vocabulary.Vocabulary(sentis, size=Z)
    return vocab

def get_train_test_sents(corpus, split=0.8, shuffle=False):
    """Get train and test sentences."""
    sentences = np.array(corpus.sents(), dtype=object)
    fmt = (len(sentences), sum(map(len, sentences)))
    print "Loaded %d sentences (%g tokens)" % fmt

    if shuffle:
        rng = np.random.RandomState(shuffle)
        rng.shuffle(sentences)  # in-place
    split_idx = int(split * len(sentences))
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]

    fmt = (len(train_sentences), sum(map(len, train_sentences)))
    print "Training set: %d sentences (%d tokens)" % fmt
    fmt = (len(test_sentences), sum(map(len, test_sentences)))
    print "Test set: %d sentences (%d tokens)" % fmt

    return train_sentences, test_sentences

def preprocess_sentences(sentences, vocab):
    # Add sentence boundaries, canonicalize, and handle unknowns
    words = ["<s>"] + flatten(s + ["<s>"] for s in sentences)
    words = [canonicalize_word(w, wordset=vocab.word_to_id)
             for w in words]
    return np.array(vocab.words_to_ids(words))

##
# Use this function
def load_corpus(name, sname, split=0.8, V=10000, Z=63, shuffle=False):
    """Load a named corpus and split train/test along sentences."""
    corpus = get_corpus(name)
    senti = [int(line[:-2]) if len(line[:-2])!=0 else int(line) for line in open('./senti.txt')]
    vocab = build_vocab(corpus, V)
    vocab_senti = build_vocab_senti(senti, Z)
    train_sentences, test_sentences = get_train_test_sents(corpus, split, shuffle)
    train_ids = preprocess_sentences(train_sentences, vocab)
    split_idx = int(split * len(senti))
    train_sids = vocab.words_to_ids(senti[:split_idx])
    test_ids = preprocess_sentences(test_sentences, vocab)
    test_sids = vocab.words_to_ids(senti[split_idx:])
    return vocab, train_ids, train_sids, test_ids, test_sids

##
# Use this function
def batch_generator(ids, batch_size, max_time):
    """Convert ids to data-matrix form."""
    # Clip to multiple of max_time for convenience
    clip_len = ((len(ids)-1) / batch_size) * batch_size
    input_w = ids[:clip_len]     # current word
    target_y = ids[1:clip_len+1]  # next word
    # Reshape so we can select columns
    input_w = input_w.reshape([batch_size,-1])
    target_y = target_y.reshape([batch_size,-1])

    # Yield batches
    for i in xrange(0, input_w.shape[1], max_time):
	yield input_w[:,i:i+max_time], target_y[:,i:i+max_time]


