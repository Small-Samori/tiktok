from utils_lda import *
import pickle, os
import gensim
import pandas as pd

def main():
    data_name = "combined_comments_3"

    data_dropped = pd.read_csv(f"./{data_name}/df_dropped.csv")
    
    with open(f"./{data_name}/lemmatized_data_text_dropped.pickle", "rb") as f:
        lemmatized_data_text_dropped = pickle.load(f)
    
    with open(f"./{data_name}/id2word.pickle", "rb") as f:
        id2word = pickle.load(f)
    
    with open(f"./{data_name}/corpus.pickle", "rb") as f:
        corpus = pickle.load(f)

    home = os.getenv("HOME")
    os.environ.update({'MALLET_HOME':r'/home/users/iasamori/tiktok/mallet-2.0.8/'})
    mallet_path = f"{home}/tiktok/mallet-2.0.8/bin/mallet"
    
    model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=65, id2word=id2word, random_seed=43)

    topics = model.show_topics(num_topics=66, num_words=15, formatted=False)
    generate_all_word_clouds(topics, data_name=data_name)
    
    get_importance_sentence(topic_model=model, corpus=corpus, 
                            data_text_lemmatized=lemmatized_data_text_dropped, 
                            data_dropped=data_dropped, topk=10, data_name=data_name)


if __name__ == "__main__":
    main()
    