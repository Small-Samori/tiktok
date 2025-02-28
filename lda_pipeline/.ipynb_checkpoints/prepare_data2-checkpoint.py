from utils_data import *
import pandas as pd
import os, pickle
import gensim.corpora as corpora


def main():
    data_path = "./rd_tt_combined_2.csv"
    csv_name = data_path.split("/")[-1]
    save_dir = csv_name.split(".")[0]
    save_path = f"./{save_dir}"
    
    df = pd.read_csv(data_path)
    tokenized_data_text = process_and_tokenize(df.message)
    
    lemmatized_data_text = ngram_and_lemmatize(tokenized_data_text)
    
    ids_to_drop = [i for i in range(len(lemmatized_data_text)) if len(lemmatized_data_text[i]) == 0]
    lemmatized_data_text_dropped = [i for i in lemmatized_data_text if len(i) != 0]
    
    df_dropped = df.drop(ids_to_drop).reset_index()
    df_dropped = df_dropped.drop(['index'], axis=1)

    id2word = corpora.Dictionary(lemmatized_data_text_dropped)
    corpus = [id2word.doc2bow(text) for text in lemmatized_data_text_dropped]

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    df_dropped.to_csv(f"{save_path}/df_dropped.csv", index=False)
    
    with open(f"{save_path}/lemmatized_data_text_dropped.pickle", "wb") as f:
        pickle.dump(lemmatized_data_text_dropped, f)
    
    with open(f"{save_path}/id2word.pickle", "wb") as f:
        pickle.dump(id2word, f)
    
    with open(f"{save_path}/corpus.pickle", "wb") as f:
        pickle.dump(corpus, f)
        
    # print(len(df), len(tokenized_data_text))

if __name__ == "__main__":
    main()