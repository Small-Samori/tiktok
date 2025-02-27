from lda_utils import *
import gensim
from tqdm import tqdm
import pickle, os

def main():
    save_path = "./combined_comments_4"

    with open(f"{save_path}/lemmatized_data_text_dropped.pickle", "rb") as f:
        lemmatized_data_text_dropped = pickle.load(f)
    
    with open(f"{save_path}/id2word.pickle", "rb") as f:
        id2word = pickle.load(f)
    
    with open(f"{save_path}/corpus.pickle", "rb") as f:
        corpus = pickle.load(f)

    nums, coherence_values = compute_coherence_values(id2word, corpus, lemmatized_data_text_dropped, limit=605)
    print(nums, coherence_values)

    with open(f"{save_path}/exp_results.pickle", "wb") as f:
        pickle.dump([nums, coherence_values], f)

if __name__ == "__main__":
    main()