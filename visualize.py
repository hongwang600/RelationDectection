import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot


# visualize using PCA
def visualize_PCA(X, names):
    start_x = -2
    start_y = 1
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1], 1)
    for i, text in enumerate(names):
        if abs(result[i,0]-start_x) < 0.2 and abs(result[i,1]-start_y) < 0.2:
            pyplot.annotate(text, (result[i,0], result[i,1]))
    pyplot.show()
    '''
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()
    '''

def read_relation_name(file_name):
    ret_names = []
    with open(file_name) as file_in:
        for line in file_in:
            ret_names.append(line[:-1])
    return ret_names

def read_embedding(file_name):
    ret_np = []
    with open(file_name) as file_in:
        for line in file_in:
            ret_np.append(np.fromstring(line[1:-2], dtype=float, sep=','))
    #print(ret_np)
    return np.asarray(ret_np)

if __name__ == "__main__":
    relation_embeddings = read_embedding('bert_feature.txt')
    relation_names = read_relation_name('question_relation.txt')
    print(len(relation_embeddings), len(relation_names))
    num_samples_2_visual = len(relation_embeddings)
    visualize_PCA(relation_embeddings[:num_samples_2_visual],
                  relation_names[:num_samples_2_visual])
