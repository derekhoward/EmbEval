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


gene2go = download_ncbi_associations()
objanno = Gene2GoReader(gene2go, taxids=[9606], go2geneids=True)
go2geneIDs = objanno.get_goid2dbids(objanno.associations)


def distance_df(emb_df, metric='euclidean'):
    if metric == 'euclidean':
        dist = euclidean_distances(emb_df)
    elif metric == 'cosine':
        dist = cosine_similarity(emb_df)
        
    dist = pd.DataFrame(dist)
    dist.index = emb_df.index
    dist.columns = emb_df.index
    
    return dist


def get_proportion_first_match(emb_df, metric='euclidean'):
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

    The input pandas data frame has only two columns, the first is the 
    low level id (image, sample, or probe_id) and the second is the 
    high level mapping (gene, region, donor). The target_value argument can
    be set to np.nan. 
    
    The output will be a matrix sized the number of low level ID's squared. 
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
        
    #dist = euclidean_distances(emb)
    #dist = pd.DataFrame(dist)
    #dist.index = emb.index
    #dist.columns = emb.index
    dist = distance_df(emb_df)
    dist.index.name = 'probe_id'
    np.fill_diagonal(dist.values, float('inf'))
    dist.drop('#na#', inplace=True)
    dist.drop('#na#', axis=1, inplace=True)
    dist = dist.sort_index(axis=0).sort_index(axis=1)
    mask = mask.sort_index(axis=0).sort_index(axis=1)
    
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

    auc = metrics.roc_auc_score(y_true, y_score)
    
    return auc


def get_GO_presence_labels(genes_of_interest, min_GO_size=200, max_GO_size=300, threshold_present=0.01, verbose=False):
    """
    Transforms a list of genes of interest into a dataframe where columns represent GO-groups and rows are TRUE/FALSE presence of gene in the GO group.
    
    genes_of_interest must be iterable of entrez_gene_ids
    """
    genes = pd.Series(genes_of_interest)
    go_group_presence = {}

    for GO in go2geneIDs:
        gene_ids = go2geneIDs[GO]
        if (len(gene_ids) > min_GO_size) & (len(gene_ids) < max_GO_size):
            in_go_group_vector = genes.isin(gene_ids)
            percent_present = in_go_group_vector.sum() / len(in_go_group_vector)                
            if percent_present >= threshold_present:
                go_group_presence[GO] = in_go_group_vector
            if verbose:
                print(f'{GO} {percent_present:.4f}')
                if percent_present < threshold_present:
                    print(f'GO group: {GO} not included')


    result = pd.DataFrame(go_group_presence)
    result.index = genes
    result.index.name = 'entrezgene'
    return result


def merge_embedding_with_GO_labels(emb_df, GO_df):
    # emb_df has index of gene_symbol
    # GO_df has index of entrezgene
    
    # output should be embedding_df with gene_symbol as index,
    # embedding cols should be prefixed with emb_, while potential GO presence columns are prefixed with GO
    
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
    
    print(f'There are {y.shape[1]} GO groups that will be evaluated.')
    
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
            print(f"Fold:{i} F1:{f1} AUC:{auc}")
            GO_SCORES.append(measures)
            
    return pd.DataFrame(GO_SCORES)
