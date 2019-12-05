from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from scipy import spatial
from tqdm import tqdm

from text_utils import PATH, SEED_PAIRS, MODEL, RANKED_PAIRS

seed_pairs = pd.read_csv(PATH+SEED_PAIRS, sep='\t')
model = Word2Vec.load(PATH+MODEL)

def extract_linguistic_code_component(seed_pairs, model):
    """
    Return the linguistic code component
    code_component = average(x - y)
    
    seed_pairs -- manually compiled seed pairs (DataFrame)
    model -- Word2Vec model built earlier (gensim.models.word2vec.Word2Vec)
    """
    x_words = list(seed_pairs['x'])
    y_words = list(seed_pairs['y'])
    
    x_vector = np.zeros((model.wv.vector_size, ))
    y_vector = np.zeros((model.wv.vector_size, ))
    
    for i in range(len(seed_pairs)):
        x_vector += model.wv[x_words[i]]
        y_vector += model.wv[y_words[i]]
    
    x_vector /= len(seed_pairs)
    y_vector /= len(seed_pairs)
    
    code_component = x_vector - y_vector
    
    return code_component

def get_threshold(seed_pairs, model):
    """
    Get threshold out of seed similarities
    Threshold is lower quartile of all seed pair similarities
    
    seed_pairs -- manually compiled seed pairs (DataFrame)
    model -- Word2Vec model built earlier (gensim.models.word2vec.Word2Vec)
    """
    seed_similarities = [model.wv.similarity(seed_pairs['x'][i], seed_pairs['y'][i])
                    for i in range(len(seed_pairs))]
    
    return np.percentile(seed_similarities, 25)

def filter_candidate_pairs(model, threshold):
    """
    Filter candidate pairs having cosine similarities greater than the threshold
    
    model -- Word2Vec model built earlier (gensim.models.word2vec.Word2Vec)
    threshold -- threshold value (float)
    """
    vocab = list(model.wv.vocab.keys())[:10000]
    candidate_pairs = []
    for i in tqdm(range(len(vocab))):
        for j in range(i+1, len(vocab)):
            if model.wv.similarity(vocab[i], vocab[j]) > threshold:
                candidate_pairs.append(tuple([vocab[i], vocab[j]]))
                
    return candidate_pairs

def rank_candidate_pairs(code_component, model, candidate_pairs):
    """
    Rank candidate pairs using score
    score(w1, w2) = cos(code_component, (w1-w2))
    
    code_component -- Linguistic code component (numpy array)
    model -- Word2Vec model built earlier (gensim.models.word2vec.Word2Vec)
    candidate_pairs -- Filtered candidate pairs (list)
    """
    score_df = pd.DataFrame(columns=['x', 'y', 'score'])
    candidate_x = [pair[0] for pair in candidate_pairs]
    candidate_y = [pair[1] for pair in candidate_pairs]
    scores = []
    for i in range(len(candidate_pairs)):        
        scores.append(1 - spatial.distance.cosine(code_component, 
                (model.wv[candidate_pairs[i][0]] - model.wv[candidate_pairs[i][1]])))
        
    score_df['x'] = candidate_x
    score_df['y'] = candidate_y
    score_df['score'] = scores
    
    score_df.sort_values(by='score', ascending=False, inplace=True)
    return score_df
    
def get_ranked_pairs():
    """
    Main function to obtain ranked word equivalence pairs
    Writes the ranked pairs to a tsv file [RANKED_PAIRS]
    """
    code_component = extract_linguistic_code_component(seed_pairs, model)
    threshold = get_threshold(seed_pairs, model)
    candidate_pairs = filter_candidate_pairs(model, threshold)
    ranked_pairs = rank_candidate_pairs(code_component, model, candidate_pairs)
    
    ranked_pairs.to_csv(RANKED_PAIRS, sep='\t', index=False)
    print('-------------Ranked pairs written to file----------------')