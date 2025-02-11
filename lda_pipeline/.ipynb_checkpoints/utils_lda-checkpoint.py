from gensim.models import CoherenceModel
import gensim
from tqdm import tqdm
import pandas as pd
import pickle, os
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image


def compute_coherence_values(id2word, corpus, texts, limit, start=5, step=10):
    home = os.getenv("HOME")
    os.environ.update({'MALLET_HOME':r'/home/users/iasamori/tiktok/mallet-2.0.8/'})
    mallet_path = f"{home}/tiktok/mallet-2.0.8/bin/mallet"
    
    coherence_values = []
    nums = []
    for num_topics in tqdm(range(start, limit, step)):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word, random_seed=43)
        
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        
        nums.append(num_topics)
        
    return nums, coherence_values

def make_wordcloud(word_dict, save_dir, topic_id=0, mask_path="oval_big_mask.png"):
    save_path = f"{save_dir}/topic_{topic_id}.png"
    mask = np.array(Image.open(mask_path))
    
    weights = list(word_dict.values())
    norm = Normalize(vmin=min(weights), vmax=max(weights))
    
    blue_cmap = LinearSegmentedColormap.from_list("shades_of_blue", ["#4545d8", "#080875"])

    def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        weight = word_dict[word]
        color = blue_cmap(norm(weight))
        return f"rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})"

    wordcloud = WordCloud(
        mask=mask,
        font_path=matplotlib.font_manager.findfont("DejaVu Sans"),  # Use sans-serif font
        width=800,
        height=400,
        background_color="white",
        prefer_horizontal=1.0,
        relative_scaling=0.5,  # Adjust scaling for better word size distribution
        max_words=100,  # Limit the maximum number of words
        random_state=42,
    ).generate_from_frequencies(word_dict)
    
    # Apply the custom coloring function
    wordcloud.recolor(color_func=blue_color_func)

    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Topic {topic_id}")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path)
    plt.show()

def generate_all_word_clouds(topics, data_name="combined_comments_3"):
    save_dir = f"./{data_name}/model_{len(topics)}topics_wordclouds"
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    for topic in tqdm(topics, desc="Generating Wordclouds ..."):
        mask = np.array(Image.open('oval_big_mask.png'))
    
        topic_id = topic[0] + 1
        topic_list = topic[1]
        
        # Create a dictionary of words and their frequencies
        word_dict = {i[0]: i[1] for i in topic_list}
        make_wordcloud(word_dict, save_dir, topic_id)


def get_importance_sentence(topic_model, corpus, 
                            data_text_lemmatized, data_dropped, topk=10, data_name="combined_comments_3"):
    topic_distribtuion = topic_model[corpus]

    corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in topic_distribtuion]
    
    corpus_topic_df = pd.DataFrame()
    corpus_topic_df["Dominant Topic"] = [item[0]+1 for item in corpus_topics]
    corpus_topic_df["Contribution (pct)"] = [round(item[1]*100, 2) for item in corpus_topics]
    corpus_topic_df["BoW"] = data_text_lemmatized
    corpus_topic_df["message"] = data_dropped["message"]

    topk_df = corpus_topic_df.groupby("Dominant Topic").apply(lambda topic_set: 
                                                (topic_set.sort_values(by=["Contribution (pct)"], ascending=False).iloc[:topk])).reset_index(drop=True)

    n_topics = topk_df["Dominant Topic"].nunique()
    
    topk_df.to_csv(f"./{data_name}/topk_importance_{n_topics}topics.csv", index=False)
    
    # return topk_df