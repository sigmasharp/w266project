import re
import time
import itertools
import numpy as np
import pandas as pd
from IPython.display import display

def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
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
import nltk
import vocabulary
from nltk.tokenize import word_tokenize
import unicodedata

def get_sents(name="brown"):
    if name=='brown':
        return nltk.corpus.__getattr__(name)

    return [word_tokenize(unicode(s.decode('utf-8').strip())) for s in open('./'+name)]

def build_vocab(sents, V=10000):
    words = flatten(sents)
    token_feed = (canonicalize_word(w) for w in words)
    vocab = vocabulary.Vocabulary(token_feed, size=V)
    return vocab

def get_train_test_sents(sents, split=0.8, shuffle=False):
    """Get train and test sentences."""
    sentences = np.array(sents, dtype=object)
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

def preprocess_ssentences(sentences, ssentences, vocab):
    #build corresponding sentiments
    # Add sentence boundaries, canonicalize, and handle unknowns
    #words = ["<s>"] + flatten(s + ["<s>"] for s in sentences)
    #words = [canonicalize_word(w, wordset=vocab.word_to_id)
    #         for w in words]
    idx = 0
    words = [int(ssentences[idx][0])]
    for s in sentences:
        ss = int(ssentences[idx][0])
        for w in s:
            words.append(ss)
        words.append(ss)
        idx = idx + 1
    
    #print vocab.ordered_words()
    return np.array(vocab.words_to_ids(words))

##
# Use this function
def load_corpus(name, sname, split=0.8, V=10000, Z=63, shuffle=False):
    """Load a named corpus and split train/test along sentences."""
    sents = get_sents(name)   
    vocab = build_vocab(sents, V)
    ssents = get_sents(sname)
    svocab = vocabulary.SVocabulary()
    #print svocab.ordered_words()
    #print svocab.words_to_ids(svocab.ordered_words())
    train_sentences, test_sentences = get_train_test_sents(sents, split, shuffle)
    train_ids = preprocess_sentences(train_sentences, vocab)
    test_ids = preprocess_sentences(test_sentences, vocab)
    train_ssentences, test_ssentences = get_train_test_sents(ssents, split, shuffle)
    train_sids = preprocess_ssentences(train_sentences, train_ssentences, svocab)
    test_sids = preprocess_ssentences(test_sentences, test_ssentences, svocab)
    return vocab, train_ids, train_sids, test_ids, test_sids

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


