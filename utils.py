from pathlib import Path
import pandas as pd
from fastai.collab import CollabDataBunch
from fastai.collab import collab_learner
from fastai.torch_core import to_np


def genesymbols_2_entrezids(genelist):
    """
    Transform list of gene symbols to entrez_ids and returns a tuple of dataframes with results
    """ 
    # should check that genelist input does not have 'na' values
    probes_file = pd.read_csv('./data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv', usecols=['gene_symbol', 'entrez_id']).drop_duplicates()
    has_entrez = probes_file[probes_file.gene_symbol.isin(genelist)]
    has_entrez = has_entrez.drop_duplicates().dropna(subset=['entrez_id'])
    
    return has_entrez


def convert_probe_emb_to_gene_emb(probe_emb):
    """Convert embedding with probe_ids for index to gene symbols by averaging probes for same gene symbol.

    Args:
        probe_emb (DataFrame): embedding with index of probe_ids

    Returns:
        gene_emb (DataFrame): embedding with index of gene_symbols
    """
    all_genes = pd.read_csv('./data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv')
    
    probe2gene = all_genes[all_genes.probeset_name.isin(probe_emb.index)].loc[:, ['probeset_name', 'gene_symbol']]
    # remove probes for 'A_' and 'CUST_' gene_symbols
    probe2gene = probe2gene[~((probe2gene.gene_symbol.str.startswith('A_')) | (probe2gene.gene_symbol.str.startswith('CUST_')))]
    
    gene_emb = probe_emb.merge(probe2gene, left_index=True, right_on='probeset_name').drop('probeset_name', axis=1).groupby('gene_symbol').mean()
    
    
    return gene_emb.drop('na')