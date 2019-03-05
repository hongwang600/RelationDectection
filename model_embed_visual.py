import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
from data import gen_data, read_origin_relation
from config import CONFIG as conf

bert_feature_file = conf['bert_feature_file']


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

def compute_rel_embed(training_data, relation_names=None):
    que_rel_embeddings = read_embedding(bert_feature_file)
    rel_indexs = {}
    for i, sample in enumerate(training_data):
        rel = sample[0]
        if rel not in rel_indexs:
            rel_indexs[rel] = [i]
        else:
            rel_indexs[rel].append(i)
    '''
    for rel in rel_indexs:
        print(all_relations[rel])
        for index in rel_indexs[rel]:
            print(relation_names[index])
        break
        '''
    rel_embed = {}
    for rel in rel_indexs:
        que_rel_embeds = [que_rel_embeddings[i] for i in rel_indexs[rel]]
        rel_embed[rel] = np.mean(que_rel_embeds, 0)
    rel_ids = rel_embed.keys()
    if relation_names is not None:
        rel_names = [relation_names[rel_indexs[i][0]].split('|||')[1]
                     for i in rel_ids]
        rel_embed_value = np.array(list(rel_embed.values()))
        return rel_names, rel_embed_value, rel_embed
    else:
        return rel_embed

def read_model_embeds(file_name):
    rels = []
    ret_np = []
    is_first_line = True
    with open(file_name) as file_in:
        for line in file_in:
            if is_first_line:
                rels = np.fromstring(line[1:-2], dtype=int, sep=',')
                is_first_line = False
            else:
                ret_np.append(np.fromstring(line[1:-2], dtype=float, sep=','))
    #print(ret_np)
    return rels, np.asarray(ret_np)

# visualize using PCA
def visualize_PCA(X, names, draw_text=True):
    start_x = -2
    start_y = 1
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1], 1)
    if draw_text:
        for i, text in enumerate(names):
        #if abs(result[i,0]-start_x) < 0.2 and abs(result[i,1]-start_y) < 0.2:
            pyplot.annotate(text, (result[i,0], result[i,1]))
    pyplot.show()
    '''
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()
    '''

if __name__ == "__main__":
    for num_file in list(range(20)):
        file_name = 'model_embed/embed'+str(num_file)+'.txt'
        rels, rel_embeds = read_model_embeds(file_name)
        all_relation_names = read_origin_relation()
        rel_names = [all_relation_names[i] for i in rels]
        visualize_PCA(rel_embeds, rel_names, False)
        #num_samples_2_visual = len(relation_embeddings)
        #visualize_PCA(relation_embeddings[:num_samples_2_visual],
        #              relation_names[:num_samples_2_visual])
