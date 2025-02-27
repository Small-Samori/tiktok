from utils_embeddings import *
import gensim
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random, pickle


def get_tsne(topic_emb_dict):
    tsne_model = TSNE(n_components=2, random_state=42)
    reduced_emb = tsne_model.fit_transform(np.array(list(topic_emb_dict.values())))
    return reduced_emb

def get_optimal_distance(linkage_matrix, reduced_emb,  data_name='combined_comments_3', plot=True):
    test_ds = np.linspace(0.3,1,80)
    scores = []
    for test_d in test_ds:
        clusters = fcluster(linkage_matrix, test_d, criterion='distance')
        # print(clusters)
        score = silhouette_score(reduced_emb, clusters)
        scores.append(score)
        
    max_index = scores.index(max(scores[:40]))
    if plot:
        plt.plot(test_ds, scores)
        plt.scatter(test_ds[max_index], scores[max_index], color='red', marker='*')
        plt.ylabel('Silhouette Score')
        plt.xlabel('Cut off Distance')
        plt.grid()
        plt.savefig(f"./{data_name}/silhoutte_score_experiment.png")
        
    return round(test_ds[max_index], 2)

def plot_dendogram(linkage_matrix, distance, labels, data_name='combined_comments_3'):
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, 
               color_threshold=distance,
               labels = labels
              )
    plt.xlabel('Topic'); plt.ylabel('Distance')
    plt.savefig(f"./{data_name}/dendogram.png")
    # plt.show()

def get_color_list(n, color_seed=42):
    unwanted_colors = ['white', 'whitesmoke', 'floralwhite','ghostwhite', 'mintcream', 'azure', 'lavenderblush', 'seashell', 'linen', 'ivory','aliceblue']
    all_colors = list(mcolors.CSS4_COLORS.keys())
    all_colors = [i for i in all_colors if i not in unwanted_colors]

    random.seed(color_seed)
    selected_colors = random.sample(all_colors, n)
    return selected_colors

def plot_tsne(clusters, reduced_emb, plot_labels, color=True, data_name='combined_comments_3', color_seed=42):
    color_list = get_color_list(max(clusters),color_seed)
    colors = [color_list[i-1] for i in clusters]
    
    plt.figure(figsize=(10, 8), dpi=120)
    if color:
        scatter = plt.scatter(reduced_emb[:,0], reduced_emb[:,1], c=colors)
    else:
        scatter = plt.scatter(reduced_emb[:,0], reduced_emb[:,1])
    
    for i in range(len(plot_labels)):
        plt.text(reduced_emb[:,0][i]+0.05, reduced_emb[:,1][i], plot_labels[i], fontsize=8)
    
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.savefig(f"./{data_name}/tsne.png")
    # plt.show()


def main():
    home = os.getenv("HOME")
    os.environ.update({'MALLET_HOME':r'/home/users/iasamori/tiktok/mallet-2.0.8/'})
    mallet_path = f"{home}/tiktok/mallet-2.0.8/bin/mallet"
    
    data_name = "combined_comments_3"
    num_topics = 65
    
    with open(f"./{data_name}/id2word.pickle", "rb") as f:
        id2word = pickle.load(f)
        
    with open(f"./{data_name}/corpus.pickle", "rb") as f:
        corpus = pickle.load(f)

    print(f"Building LDA model with {num_topics} topics ...")
    optimal_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word, random_seed=43)

    print(f"Getting embeddings")
    filtered_topics = get_filtered_topics(optimal_model)
    topic_emb_small_dict, topic_emb_large_dict = get_topic_emb_dict(filtered_topics)
    
    print(f"Performing TSNE")
    plot_labels = list(topic_emb_large_dict.keys())
    reduced_emb = get_tsne(topic_emb_large_dict)
    
    print(f"Performing Clustering")
    df = pd.DataFrame(np.array(list(topic_emb_large_dict.values())))
    dist_matrix = pdist(df)
    linkage_matrix = linkage(dist_matrix, method='ward')
    
    # optimal_distance = get_optimal_distance(linkage_matrix, reduced_emb, data_name=data_name, plot=True)
    optimal_distance = get_optimal_distance(linkage_matrix, np.array(list(topic_emb_large_dict.values())), data_name=data_name, plot=True)
    
    # plot_dendogram(linkage_matrix, optimal_distance, labels=plot_labels, data_name=data_name)
    plot_dendogram(linkage_matrix, 0, labels=plot_labels, data_name=data_name)
    
    clusters = fcluster(linkage_matrix, optimal_distance, criterion='distance')
    np.save(f"./{data_name}/clusters.npy", clusters)
    
    # plot_tsne(clusters, reduced_emb, plot_labels, color=True, data_name=data_name, color_seed=10)
    plot_tsne(clusters, reduced_emb, plot_labels, color=False, data_name=data_name, color_seed=10)


if __name__ == "__main__":
    main()















