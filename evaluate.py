import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
import utils

import os


gene2go = download_ncbi_associations()
objanno = Gene2GoReader(gene2go, taxids=[9606], go2geneids=True)
go2geneIDs = objanno.get_goid2dbids(objanno.associations)  # this is a dict. Keys are GO IDs, values are gene_IDs of the genes that are associated to that GO term



def distance_df(emb_df, metric='euclidean'):
    """Creates a distance matrix for a given embedding DataFrame.

    Args:
        emb_df (DataFrame): A DataFrame of shape (n_probes, n_features)
        metric (str, optional): Distance metric, defaults to 'euclidean'. Can also compute cosine similarity.

    Returns:
        dist (DataFrame): A square DataFrame of shape (n_probes, n_probes)
    """
    if metric == 'euclidean':
        #dist = euclidean_distances(emb_df)
        #dist = euclidean_distances(emb_df, emb_df)
        dist = euclidean_distances(emb_df.iloc[:, 1:], emb_df.iloc[:, 1:])
    elif metric == 'cosine':
        dist = cosine_similarity(emb_df)
        
    dist = pd.DataFrame(dist)
    dist.index = emb_df.index
    dist.columns = emb_df.index
    
    return dist


def get_proportion_first_match(emb_df, metric='euclidean'):
    """Operates on probe embedding and checks to see if nearest probe judged by distance metric is of another probe for the same gene.

    Args:
        emb_df (pd.DataFrame): A DataFrame of shape (n_samples, n_features)
        metric (str, optional): Distance metric, defaults to 'euclidean'. Can also compute cosine similarity.

    Returns:
        float: proportion of probes that match to another probe of the same gene
    """
    dist = distance_df(emb_df, metric=metric)
    
    if metric == 'euclidean':
        np.fill_diagonal(dist.values, float('inf'))
        closest_indexes = dist.idxmin(axis=1, skipna=True).reset_index()
    elif metric == 'cosine':
        np.fill_diagonal(dist.values, float('0'))
        closest_indexes = dist.idxmax(axis=1, skipna=True).reset_index()

    closest_indexes.columns = ['probe_id', 'neighbor']
    closest_indexes['gene_id'] = closest_indexes.probe_id.map(probe_map)
    closest_indexes['nearest_gene'] = closest_indexes.neighbor.map(probe_map)
    proportion_closest_match = closest_indexes[closest_indexes.gene_id == closest_indexes.nearest_gene].shape[0] / closest_indexes.shape[0]
    
    return proportion_closest_match


def get_closest_probes(emb_df, probe_id, metric='euclidean'):
    dist = distance_df(emb_df, metric=metric)
    dist.index.name = 'probe_id'
    
    result = dist[dist.index == probe_id].iloc[0, :].sort_values()
    result = result.reset_index()
    result['gene'] = result.probe_id.map(probe_map)
    
    return result


def create_diagonal_mask(low_to_high_map, target_value = 1):
    """Create a block diagonal mask matrix from the input mapping.

    Args:
        low_to_high_map (pd.DataFrame): df has only two columns, the first is the 
    low level id (image, sample, or probe_id) and the second is the 
    high level mapping (gene, region, donor).

        target_value (int, optional): Defaults to 1. Can
    be set to np.nan. 

    Returns:
        relationship_df (pd.DataFrame): a matrix sized the number of low level ID's squared. 
    The column and row order will have to be rearranged to match your distance matrix.
    """
    low_to_high_map.drop_duplicates()
    grouped = low_to_high_map.groupby(low_to_high_map.columns[1])
    ordered_low_level_names = list()
    current_diagonal_location = 0
    group_matrices = []
    for name, group in grouped:
        group_size = group.shape[0]
        # build up row/col names, order doesn't matter within a group = they are all equal
        ordered_low_level_names = ordered_low_level_names + group.iloc[:,0].tolist()
        # set the diagonal matrix to be the target value
        single_group_matrix = np.full(shape = (group_size,group_size), fill_value = target_value)
        group_matrices.append(single_group_matrix)
    # add the individual matrices along the diagonal
    relationship_matrix = scipy.linalg.block_diag(*group_matrices)
    # convert to pandas dataframe and set names
    relationship_df = pd.DataFrame(relationship_matrix, columns = ordered_low_level_names, index = ordered_low_level_names)
    return relationship_df


def calc_probe_match_auc(emb_df, mask, probe_map='default', metric='euclidean'):
    """Calculates AUC where matches are for different probes of same gene symbol.

    Args:
        emb_df (pd.DataFrame): A DataFrame of shape (n_samples, n_features)
        probe_map (str, optional): Mapping of probes to gene symbols. Default is from Allen fetal brain.
        metric (str, optional): Defaults to 'euclidean'.
        
    Returns:
        auc (float)
    """
    
    if probe_map == 'default':
        probe_ids = pd.read_csv('./data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv', usecols=['probeset_name', 'gene_symbol'])
        probe_ids = probe_ids.set_index('probeset_name').to_dict()['gene_symbol']
        
    elif probe_map == 'reannotator':
        # the following is to map probes in the same manner as was done while training NN/embeddings
        probe_ids = pd.read_table('./data/raw/gene_symbol_annotations/AllenInstitute_custom_Agilent_Array.txt', sep='\t')
        probe_ids = probe_ids.rename(columns={'#PROBE_ID': 'probe_id', 
                      'Gene_symbol': 'gene_symbol'}).loc[:, ['probe_id', 'gene_symbol']]
        probe_ids.gene_symbol = probe_ids.gene_symbol.str.split(';').str[0]
        probe_ids = probe_ids.set_index('probe_id').to_dict()['gene_symbol']
    else:
        raise ValueError("Error: specify probe_map as either 'default' or 'reannotator'.")
        

    dist = distance_df(emb_df)
    dist.index.name = 'probe_id'
    np.fill_diagonal(dist.values, float('inf'))
    #dist.drop('#na#', inplace=True)
    #dist.drop('#na#', axis=1, inplace=True)
    dist = dist.sort_index(axis=0).sort_index(axis=1)
    #mask = mask.sort_index(axis=0).sort_index(axis=1)
    
    values = dist.values
    i, j = np.tril_indices_from(values, k=-1)
    pairwise_dists = pd.DataFrame.from_dict({'probe_id1':dist.index[i], 
                                             'probe_id2': dist.columns[j],
                                             'distance': values[i,j]})
    
    pairwise_dists['gene1'] = pairwise_dists['probe_id1'].map(probe_ids)
    pairwise_dists['gene2'] = pairwise_dists['probe_id2'].map(probe_ids)
    pairwise_dists['same_gene'] = pairwise_dists['gene1'] == pairwise_dists['gene2']
    
    y_score = pairwise_dists.distance 
    y_true = pairwise_dists.same_gene

    if metric == 'euclidean':
        auc = 1 - metrics.roc_auc_score(y_true, y_score)
    else:
        auc = metrics.roc_auc_score(y_true, y_score)
    
    return auc


def get_GO_presence_labels(genes_of_interest, min_GO_size=200, max_GO_size=300, threshold_present=0.01, verbose=False):
    """Creates a dataframe of GO-group presence for a list of genes.
    
    Args:
        genes_of_interest : must be iterable of entrez_gene_ids
        min_GO_size (int, optional): Min num of genes in GO group to be included. Defaults to 200.
        max_GO_size (int, optional): Max num of genes in GO group to be included. Defaults to 300.
        threshold_present (float, optional): Percent of genes in GO group that should be present in gene_list. Defaults to 0.01.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        pd.DataFrame : df where index is entrezgene, columns are GO group with TRUE/FALSE presence values.
    """
    genes = pd.Series(genes_of_interest)
    go_group_presence = {}

    for GO in go2geneIDs:
        gene_ids = go2geneIDs[GO]
        if (len(gene_ids) > min_GO_size) & (len(gene_ids) < max_GO_size):
            in_go_group_vector = genes.isin(gene_ids)
            print ("in go group vector is: ", in_go_group_vector)

            percent_present = in_go_group_vector.sum() / len(in_go_group_vector)                
            if percent_present >= threshold_present:
                go_group_presence[GO] = in_go_group_vector
            if verbose:
                #print(f'{GO} {percent_present:.4f}')
                print ("{}    {}".format(GO, percent_present))
                if percent_present < threshold_present:
                    #print(f'GO group: {GO} not included')
                    print ("Go group: {} not included".format(GO))



    print ("GO group presence dict is: ", go_group_presence)
    result = pd.DataFrame(go_group_presence)
    result.index = genes
    result.index.name = 'entrezgene'


    print ("final result is: ", result)
    return result


def merge_embedding_with_GO_labels(emb_df, GO_df):
    """Merges an gene_embedding with GO group presence df.
    
    Embedding cols are prefixed with emb_, while potential GO presence columns are prefixed with GO:

    Args:
        emb_df (pd.DataFrame): emb_df.index is gene_symbol
        GO_df (pd.DataFrame): GO_df.index is entrezgene

    Returns:
        (pd.DataFrame): Multi-index gene embedding with columns for GO presence concatenated.
    """
    # get df with gene_symbols and entrez_ids from fetal data (more updated than adult probes data)
    all_genes = pd.read_csv('./data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv')
    all_genes = all_genes[~((all_genes.gene_symbol.str.startswith('A_')) | (all_genes.gene_symbol.str.startswith('CUST_')))].gene_symbol.drop_duplicates()
    all_genes_w_entrez = utils.genesymbols_2_entrezids(all_genes)
    
    emb_df = emb_df.add_prefix('emb_')
    df = emb_df.merge(all_genes_w_entrez, left_index=True, right_on='gene_symbol')
    df = df.merge(GO_df, left_on='entrez_id', right_index=True)
    
    return df.set_index(['entrez_id', 'gene_symbol'])


def perform_GOclass_eval(embedding_df,
                          index_type='gene_symbol',
                          min_GO_size=200,
                          max_GO_size=300,
                          threshold_present=0.01,
                          n_splits=5,
                          n_jobs=-1):
    
    if index_type == 'gene_symbol':
        # convert index to entrez_id
        entrez_genelist = utils.genesymbols_2_entrezids(embedding_df.index)
        GO_df = get_GO_presence_labels(genes_of_interest=entrez_genelist.entrez_id, min_GO_size=min_GO_size, max_GO_size=max_GO_size, threshold_present=threshold_present)
    elif index_type == 'entrez_id':
        GO_df = get_GO_presence_labels(genes_of_interest=embedding_df.index, min_GO_size=min_GO_size, max_GO_size=max_GO_size, threshold_present=threshold_present)
    else:
        raise ValueError("Error: specify index type as either 'gene_symbol' or 'entrez_id'.")
    
    # merge the embedding and GO_df to ensure they have same index
    # returns a multi-index df with gene_symbol and entrez_id
    merged_df = merge_embedding_with_GO_labels(emb_df=embedding_df, GO_df=GO_df)
    X = merged_df.loc[:, merged_df.columns.str.startswith('emb_')]
    y = merged_df.loc[:, merged_df.columns.str.startswith('GO:')]
    
    #print(f'There are {y.shape[1]} GO groups that will be evaluated.')
    print ("there are ")
    
    GO_SCORES = []
    skf = StratifiedKFold(n_splits=n_splits)
    
    for GOlabel in y:
        print('--'*50)
        print(GOlabel)
        y_GO = y.loc[:, GOlabel]

        for i, (train_idx, test_idx) in enumerate(skf.split(X, y_GO)):
            model = LogisticRegression(penalty='none', n_jobs=n_jobs)
            X_train = X.iloc[train_idx, :]
            y_train = y_GO.iloc[train_idx]
            X_test = X.iloc[test_idx, :]
            y_test = y_GO.iloc[test_idx]

            model.fit(X_train, y_train)

            # Extract predictions from fitted model
            preds = model.predict(X_test)
            # probs for classes ordered in same manner as model.classes_
            # model.classes_  >>  array([False,  True])
            probas = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)

            # Get metrics for each model
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, probas[True])
            measures = {'GO_group': GOlabel,
                        'iteration': i,
                        'f1': f1, 
                        'AUC': auc}
            #print(f"Fold:{i} F1:{f1} AUC:{auc}")
            print ("Fold")
            GO_SCORES.append(measures)
            
    return pd.DataFrame(GO_SCORES)



if __name__ == "__main__":

    embed_file_name =  "validation_embeddings_image_level.csv"
    path_to_embed = os.path.join("/Users/pegah_abed/Documents/Embedding_Evaluation/embed_eval", embed_file_name)
    emb_df = pd.read_csv(path_to_embed)
    print ("number of images: ", len(emb_df))

    #dist_matrix = distance_df(emb_df)
    #print ("number of rows: ", len(dist_matrix))
    #dist_matrix.to_csv(os.path.join("/Users/pegah_abed/Documents/Embedding_Evaluation/embed_eval", "distance_matrix_3.csv"))

    get_proportion_first_match(emb_df)

