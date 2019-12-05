import time
import glob
import re
import nltk
import string
from gensim.models import Word2Vec

from text_utils import PATH, DATA, WORDS, tweet_delim, urls, punct, hash_tag, handle_tag, handle_sub

def read_data():
    """
    Read the data files
    """
    files = glob.glob(PATH+DATA)
    text = []
    for name in files:
        with open(name) as f:
            text += f.readlines()
    
    with open(PATH+WORDS) as f:
        words = f.readlines()
    
    text = [t.strip() for t in text]
    words = [w.strip() for w in words]

    return text, words

def if_valid(tweet_string, matched, words):
    """
    Check if tweet string has sufficient length, number of words, 
    and is not the delimiting string that you have used to separate tweets 
    in your corpus
    
    tweet_string -- string to be checked (string)
    matched -- a purely optional list to check which word caused this tweet
    to be added to the corpus from the scraping tool (list) [used for debugging in the scraping process]
    words -- list of input words to the scraping tool (list) [used for debugging in the scraping process]
    """
    if len(tweet_string.strip()) < 3:
        return 0
    elif len(tweet_string.split(' ')) < 2:
        return 0
    elif tweet_string == tweet_delim:
        return 0
    else:
        return 1
        for word in words:
            if word in [str.lower(s) for s in tweet_string.split(' ') if s]:
                matched.append(word)
                return 1
        return 0

def remove_url(tweet_string):
    """
    Handle urls in tweets plus misc. filtering
    
    tweet_string -- string to be checked (string)
    """
    for pattern in urls:
        tweet_string = re.sub(pattern, '', tweet_string)
    tweet_string = ''.join([a for a in tweet_string 
                            if a in string.digits+string.ascii_letters+punct])
    
    consecutivedots = re.compile(r'\.{2,}')
    tweet_string = consecutivedots.sub('', tweet_string)
    return tweet_string.strip()

def remove_hashtags(tweet_string):
    """
    Handle hashtags in tweets by regex
    Converts #GoodMorning to Good Morning
    Converts #FIFAWC to FIFAWC
    Converts #bored to bored
    
    tweet_string -- string to be checked (string)
    """
    if '#' not in tweet_string:
        return tweet_string
    for match in re.findall(hash_tag, tweet_string):
        tag = match.strip()[1:]
        if tag.islower() or tag.isupper():
            if len(tag)>4:
                if tag[-4:].isnumeric():
                    tweet_string = re.sub(match, ' '+tag[:-4]+' '+tag[-4:], tweet_string)
                else:
                    tweet_string = re.sub(match, ' '+tag, tweet_string)
            else:
                tweet_string = re.sub(match, ' '+tag, tweet_string)
        else:
            repl = ' '.join(re.findall(r'[A-Z][a-z]+[^A-Z]*', tag))
            tweet_string = re.sub(match, ' '+repl, tweet_string)

    return tweet_string.strip()

def remove_handles(tweet_string):
    """
    Handle Twitter handles in tweets by regex
    Converts @MontyPython to Monty Python
    Converts @POTUS to <hndl>
    Converts @wert to <hndl>
    
    tweet_string -- string to be checked (string)
    """
    if '@' not in tweet_string:
        return tweet_string
    for match in re.findall(handle_tag, tweet_string):
        tag = match.strip()[1:]
        if tag.islower() or tag.isupper():
            if len(tag)>4:
                if tag[-4:].isnumeric():
                    tweet_string = re.sub(match, ' '+tag[:-4]+' '+tag[-4:], tweet_string)
                else:
                    tweet_string = re.sub(match, handle_sub, tweet_string)
            else:
                tweet_string = re.sub(match, handle_sub, tweet_string)
        else:
            repl = ' '.join(re.findall(r'[A-Z][a-z]+[^A-Z]*', tag))
            tweet_string = re.sub(match, ' '+repl, tweet_string)

    return tweet_string.strip()

def preprocess(valid_text):
    """
    Calls all specific cleaning functions and returns a list of cleaned sentences
    
    valid_text -- list of text strings (list)
    """
    
    cleaned_text = [remove_url(remove_handles(remove_hashtags(remove_url(t)))) for t in valid_text]
    
    cleaned_text = list((set(cleaned_text)))

    cleaned_text = [nltk.tokenize.sent_tokenize(t) for t in cleaned_text]
    
    cleaned_text = [item for sublist in cleaned_text for item in sublist]
    
    clean_sentences = [nltk.tokenize.word_tokenize(t) for t in cleaned_text]

    mwetokenizer = nltk.MWETokenizer(separator='')
    mwetokenizer.add_mwe(('<', 'hndl', '>'))
    clean_sentences = [mwetokenizer.tokenize(t) for t in clean_sentences]
        
    new_sentences = []
    for sentence in clean_sentences:
        sentence = [str.lower(token) for token in sentence if 0<len(token)<17 and
                    token not in '@.?\n\t']
        new_sentences.append(sentence)

    return new_sentences

def train(emb_size, sg, hs, negative, min_count, n_iter, workers):
    """
    Train the Word2Vec model and save it.
    
    All arguments are identical to the ones in gensim.models.Word2Vec
    """
    text, words = read_data()
    matched = []
    valid_text = [t for t in text if if_valid(t, matched, words)]
    
    sentences = preprocess(valid_text)
    
    # Training Word2Vec model
    model = Word2Vec(sentences, size=emb_size, sg=sg, hs=hs, negative=negative, 
                     min_count=min_count, iter=n_iter, compute_loss=True, workers=workers)
    model.save(PATH+f"Models/word2vec_{int(time.time())}.model")
