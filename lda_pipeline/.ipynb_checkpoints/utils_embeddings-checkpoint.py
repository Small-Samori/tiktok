from openai import AzureOpenAI
import os, pickle
import numpy as np
import random
import dotenv
dotenv.load_dotenv()

def generate_embeddings(text, size="SMALL"): # model = "deployment_name"
    client = AzureOpenAI(
      api_key = os.getenv("AZURE_OPENAI_KEY"),  
      api_version = "2024-02-01",
      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    return client.embeddings.create(input = [text], model=os.getenv(f"DEPLOYMENT_NAME_{size}")).data[0].embedding


def download_emb(text, embedding_folder=f"{os.getenv('OAK')}/samori/tiktok/embeddings"):
    small_emb = generate_embeddings(text, size="SMALL")
    large_emb = generate_embeddings(text, size="LARGE")
    
    with open(f"{embedding_folder}/gensim_vocab_embedding_small.pkl", "rb") as f:
        emb_dict_small = pickle.load(f)
        emb_dict_small[text] = small_emb

    with open(f"{embedding_folder}/gensim_vocab_embedding_large.pkl", "rb") as f:
        emb_dict_large = pickle.load(f)
        emb_dict_large[text] = large_emb

    with open(f"{embedding_folder}/gensim_vocab_embedding_small.pkl","wb") as f:
        pickle.dump(emb_dict_small, f)

    with open(f"{embedding_folder}/gensim_vocab_embedding_large.pkl","wb") as f:
        pickle.dump(emb_dict_large, f)

    return (emb_dict_small, emb_dict_large)


def get_filtered_topics(model, threshold=0.80):
    
    # optimal_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=65, id2word=id2word, random_seed=43)
    topics = model.show_topics(num_topics=66, num_words=16861, formatted=False)
    filtered_topics = {}
    for i in range(len(topics)):
        topic_list = topics[i][1]
        word_dict = {i[0]: i[1] for i in topic_list}
        
        words = list(word_dict.keys())
        values = list(word_dict.values())
    
        cumsum = np.cumsum(values)
        n = np.argmin(cumsum <= threshold)
    
        
        filtered_dict = dict(zip(words[:n], values[:n]))
        filtered_topics[i] = filtered_dict
        
    return filtered_topics

def get_topic_emb_dict(filtered_topics):

    oak = os.getenv('OAK')
    embedding_folder = f"{oak}/samori/tiktok/embeddings"
    
    with open(f"{embedding_folder}/gensim_vocab_embedding_small.pkl", "rb") as f:
        emb_dict_small = pickle.load(f)
    
    with open(f"{embedding_folder}/gensim_vocab_embedding_large.pkl", "rb") as f:
        emb_dict_large = pickle.load(f)
    
    topic_emb_small_dict = {}
    topic_emb_large_dict = {}
    for i in range(len(list(filtered_topics.keys()))):
        topic_dict_i = filtered_topics[i]
    
        terms = list(topic_dict_i.keys())
        terms = [i.replace("_"," ") for i in terms]
        # print(terms)
        # break
        values = list(topic_dict_i.values())
        values = np.array(values) / np.sum(values)
    
        topic_emb_small = np.zeros(1536)
        topic_emb_large = np.zeros(3072)
        
        for term_id in range(len(terms)):
            try:
                topic_emb_small += values[term_id] * np.array(emb_dict_small[terms[term_id]])
                topic_emb_large += values[term_id] * np.array(emb_dict_large[terms[term_id]])
            except KeyError:
                print(terms[term_id])
                emb_dict_small, emb_dict_large = download_emb(terms[term_id])
                topic_emb_small += values[term_id] * np.array(emb_dict_small[terms[term_id]])
                topic_emb_large += values[term_id] * np.array(emb_dict_large[terms[term_id]])
    
        topic_emb_small_dict[f"{i+1}"] = topic_emb_small
        topic_emb_large_dict[f"{i+1}"] = topic_emb_large
    return topic_emb_small_dict, topic_emb_large_dict