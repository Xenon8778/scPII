import numpy as np
import pandas as pd
import scipy.sparse.linalg
import scipy.linalg
import scipy.cluster.hierarchy as sch
import networkx as nx
from .utils import div0
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

def decompose(L,
            n_genes,
            n_comps: int = None,
            explainedV: float = 1,
            verbose: bool = True):
    """
    Performing hermitian eigen decomposition on graph laplacian
    """
    if n_comps == None:
        if explainedV == 1:
            n_comps = n_genes 
            values,vectors = np.linalg.eigh(L)
            vectors = vectors[:,values > 1e-5]
            values = values[values > 1e-5]
            if verbose == True:
                print('Keeping '+ str(len(values))+' eigen values and associated vectors')
            
        else:
            values,vectors = np.linalg.eigh(L)
            vectors = vectors[:,values > 1e-5]
            values = values[values > 1e-5]
            totalVariance = sum(1/values)
            cumVariance = np.cumsum(1/values) # Calculate cummulative sum of eigenvalues
            for i,val in enumerate(cumVariance):
                if val >= explainedV*totalVariance:
                    n_comps = i
                    break
            if verbose == True:
                print(str(n_comps)+ ' components explain '+str(explainedV)+' of total variance.')
            values,vectors = np.linalg.eigh(L)
            values = values[:n_comps+1]
            vectors = vectors[:, :n_comps+1]
            vectors = vectors[:,values > 1e-5]
            values = values[values > 1e-5]
            if verbose == True:
                print('Keeping '+ str(len(values))+' eigen values and associated vectors')
    
    else:
        if L.shape[0] <= n_comps:
            values,vectors = scipy.linalg.eigh(L)
        else:
            values,vectors = scipy.linalg.eigh(L, subset_by_index=[0,n_comps])
        if verbose == True:
            print('Using ' + str(n_comps-sum(values <= 1e-5)) + ' and Dropping '+ str(sum(values <= 1e-5))+' zero eigen values and associated vectors')
        vectors = vectors[:,values > 1e-5]
        values = values[values > 1e-5]

    if verbose:
        print('DONE')
    return(values,vectors)

def computePRS(values, vectors, n_genes,
            verbose: bool = True):
    
    if verbose:
        print('Calculating Inverse Laplacian...')
    val_inv = div0(1,values)
    
    # Calculating Inverse Laplacian matrix
    cov = vectors @ np.diag(val_inv) @ vectors.T

    prs_matrix = cov**2

    # Normalizing
    ''' The normalization accounts for the topology-defined adaptability of the nodes 
    and breaks the symmetry of the covariance matrix. The row and column averages of 
    the PRS matrix give the effectiveness and the sensitivity profiles as a 
    function of gene index [1, n], respectively.'''

    if verbose == True:
        print('Normalizing...')
    norm_prs_matrix = np.zeros((n_genes, n_genes))
    self_dp = np.diag(prs_matrix)  
    self_dp = self_dp.reshape(n_genes, 1)
    re_self_dp = np.repeat(self_dp, n_genes, axis=1)
    norm_prs_matrix = div0(prs_matrix, re_self_dp)
    W = 1 - np.eye(n_genes)
    norm_prs_matrix = norm_prs_matrix*W
    if verbose == True:
        print('DONE')

    return norm_prs_matrix

def shuffle_means(X, alpha):
    # Flatten the array
    Y = X.copy()
    np.fill_diagonal(Y, np.nan)
    X_flat = Y.flatten()
    np.random.shuffle(X_flat)
    X_shuffled = X_flat.reshape(Y.shape)
    del Y
    eff_orig = np.nanmean(X_shuffled, axis=1)
    sens_orig = np.nanmean(X_shuffled, axis=0)
    
    # Log transformation and impact calculation
    impact_scores = np.log1p(eff_orig) - np.log1p(sens_orig) * alpha
    impact_scores_mm = (impact_scores - min(impact_scores)) / (max(impact_scores) - min(impact_scores))
    return impact_scores_mm

def getPvals(X,
            df,
            n: int=10000,
            alpha: float=1.0,
            n_jobs: int=1,
            verbose: bool=True):
    """
    Computes empirical p-values by shuffling the PRS matrix.

    Parameters
    ----------
    X : np.ndarray
        The PRS matrix.
    df : pd.DataFrame
        DataFrame containing the impact values.
    n : int, default=1000
        Number of shuffles to perform.
    alpha : float, default=1
        Scaling factor for sensitivity.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    verbose : bool, default=True
        Whether to print progress messages.

    Returns
    -------
    np.ndarray
        Array of empirical p-values.

    References
    ----------
    Davison, A. C., & Hinkley, D. V. (1997). Bootstrap Methods and Their Application. Cambridge University Press.
    """
    # Computes emperical Pvalues by shuffling the PRS matrix
    impacts = pd.DataFrame(np.zeros((X.shape[0],n)))
    if verbose:
        print('Computing p-values...')

    from joblib import parallel_backend

    if verbose:
        with parallel_backend('loky', n_jobs=n_jobs):
            results = Parallel()(delayed(shuffle_means)(X, alpha) for _ in tqdm(range(n)))
    else:
        with parallel_backend('loky', n_jobs=n_jobs):
            results = Parallel()(delayed(shuffle_means)(X, alpha) for _ in range(n))

    for i, result in enumerate(results):
        impacts[i] = result

    # empirical p-value formula following Davison and Hinkley
    P = (np.sum(impacts.sub(df['impact'], axis=0) > 0, axis=1) + 1) / (n + 1)
    return P

def getSummaryDF(X, G, L, 
                getPval: bool = True,
                n_genes: int = None, 
                alpha: float = 1, 
                verbose: bool = True,
                n_boot: int = 10000,
                n_jobs: int = 1):
    """
    Summarizes the PRS matrix and computes impact scores and p-values.

    Parameters
    ----------
    X : np.ndarray
        The normalized PRS matrix.
    G : nx.Graph
        The gene regulatory network graph.
    L : np.ndarray
        The Laplacian matrix of the graph.
    getPval : bool, default=True
        Whether to compute p-values.
    n_genes : int, optional
        Number of genes.
    alpha : int, default=1
        Scaling factor for sensitivity.
    verbose : bool, default=True
        Whether to print progress messages.
    n_boot : int, default=10000
        Number of bootstrap samples for p-value computation.
    n_jobs : int, default=1
        Number of parallel jobs to run.

    Returns
    -------
    pd.DataFrame
        DataFrame containing summary metrics for each gene.
    """
    # Summary DataFrame
    if verbose == True:
        print('Summarising...')
    df = pd.DataFrame()
    if n_genes is None:
        n_genes = len(G.nodes)
    df['gene_name'] = list(G.nodes)
    df['deg'] = np.diag(L)
    W = 1 - np.eye(n_genes)
    eff_orig = np.average(X, weights=W, axis=1)
    sens_orig = np.average(X, weights=W, axis=0)
    df_ = df

    # Log tranformation
    df_['eff'] = np.log1p(eff_orig)
    df_['sens'] = np.log1p(sens_orig)
    # df_['trans'] = list((nx.clustering(graph_gc)).values()) # Graph clustering
    dfc = df_[['eff','sens']] #/ df_[['eff','sens']].max()
    dfc.loc[:, 'sens'] = dfc['sens'] * alpha
    df_['impact'] = np.diff(dfc[['sens','eff']], axis = 1)
    mmscaler = MinMaxScaler()
    df_['impact'] = mmscaler.fit_transform(df_[['impact']])

    # Calculate P Values
    if getPval:
        df_['P'] = getPvals(X=X, df=df_, alpha=alpha, n=n_boot, n_jobs=n_jobs, verbose=verbose)
    
    df_ = df_.sort_values('impact', ascending=False)
    return df_

def clusterPRS(X, G, Kclusters, df, verbose):
    if verbose:
        print('Clustering...')
    row_dist = sch.distance.pdist(X, metric='seuclidean')
    row_linkage = sch.linkage(row_dist, method='ward',optimal_ordering=True)
    root_row, tree_list_row = sch.to_tree(row_linkage, True)

    col_dist = sch.distance.pdist(X.T, metric='seuclidean')
    col_linkage = sch.linkage(col_dist, method='ward',optimal_ordering=True)
    root_col, tree_list_col = sch.to_tree(col_linkage, True)
    nds_row = root_row.pre_order()
    nds_col = root_col.pre_order()

    cluster = sch.fcluster(row_linkage, Kclusters, criterion='maxclust')
    df['eff_clust'] = cluster

    cluster = sch.fcluster(col_linkage, Kclusters, criterion='maxclust')
    df['sens_clust'] = cluster

    clustered_mat = pd.DataFrame(X[:,nds_row][nds_col], index = np.array(G.nodes)[nds_row], columns= np.array(G.nodes)[nds_col])
    clustered_mat.head()
    if verbose:
        print('DONE')        
    
    return clustered_mat, df

def scPRS(X: pd.DataFrame,
        gene_names: str = None,
        Corr_cutoff: float = 0,
        alpha: float = 1,
        getPval: bool = False,
        cluster: bool = False,
        Kclusters: int = 10,
        n_comps: int = None,
        use_GC: bool = True,
        weighted: bool = True,
        explainedV: float = 0.1,
        n_boot: int = 10000,
        n_jobs: int = 1,
        verbose: bool = True
        ):
    """
    Performing perturbation response scanning on gene regulatory networks

    Parameters
    ----------
    X: pd.DataFrame
        A gene regulatory network X, expected shape = (n_genes, n_genes)
    gene_names: str, default = None
        List of genes to use for analysis
    Corr_cutoff = float, default = 0
        The minimum correlation cutoff to filter out edges with low interaction scores from input gene regulatory network
    alpha = float, default = 1
        The effect of sensitivity on impact computation.
    clusters: bool, default = False
        Whether to perform hierarchical clustering of nodes
    Kclusters: int, default = 10
        The number of effectiveness and sensitivity hierarchical clusters 
    n_comps: int, default = None
        number of PCs for covariance matrix computation. Suggested to use None for automatic selection based on explained variance, else use 20.
    use_GC = bool, default = True
        Whether to use the giant component of graph for PRS computation
    weighted: bool, default = True
        Whether to use weighted adjacency matrix for PRS computation 
    verbose: bool, default = True
        Whether to output comments

    Returns
    -------
    clustered_mat: pd.DataFrame
        A dataframe that contains PRS matrix result, expected shape = (n_genes, n_genes)
    summary_df: pd.DataFrame
        A dataframe that contains PRS summary results for each node (gene)
    graph_gc: nx.network
        A network object storing the gene regulatory network
    """

    if X.shape[0] != X.shape[1]:
        raise ValueError("Matrix must be square.")
    genes = list(gene_names)
    n_genes = len(gene_names)

    ## Reading in data as numpy array
    X = np.array(X)
    np.fill_diagonal(X,0)

    ## Binarizing using cutoff and removing unlinked nodes
    X[X <= Corr_cutoff] = 0

    ## Converting to Pandas DataFram
    df = pd.DataFrame(X, index = genes, columns= genes)
    df = df.loc[:, df.astype(bool).sum(axis=0) > 0].copy()
    df = df.loc[df.astype(bool).sum(axis = 1) > 0,:].copy()

    if verbose == True:
        print('Correlation matrix created. Shape: ', df.shape)

    edgelist = df.stack().reset_index()
    edgelist.columns = ['gene1','gene2','Val']
    edgelist = edgelist[edgelist['Val'] != 0].reset_index(drop=True)

    G = nx.from_pandas_edgelist(edgelist, source='gene1', target='gene2', edge_attr = True)
    if use_GC == True:
        # Calculating giant component - Largest connected componenet of the graph
        if verbose == True:
            print('Calculating giant components...')
        Gc = max([G.subgraph(c).copy() for c in nx.connected_components(G)], key=len)
        graph_gc = Gc
        if verbose == True:
            print('Giant component shape: ', nx.adjacency_matrix(Gc).shape)

        # Getting Laplacian matrix
        if weighted == True:
            L = nx.laplacian_matrix(graph_gc, weight='Val').todense()
        else:
            L = nx.laplacian_matrix(graph_gc, weight=None).todense()

        # Eigen decomposition
        if verbose == True:
            print('Performing eigen decomposition...')
        n_genes = L.shape[0]
        if n_genes <= 5:
                sys.exit("Too few genes")

        values, vectors = decompose(L=L, n_genes=n_genes, n_comps=n_comps,
                                    explainedV=explainedV,
                                    verbose=verbose)
        
        if len(vectors) <= 5:
            sys.exit("Error: Less than 5 Eigenpairs explain {explainedV} variance. Please increase 'explainedV'.")
        
        # PRS matric computation
        norm_prs_matrix = computePRS(values=values, n_genes=n_genes,
                                vectors=vectors, verbose=verbose)
        
        # Get summary metrics
        summDF = getSummaryDF(X=norm_prs_matrix, G=graph_gc, L=L, alpha=alpha, n_boot=n_boot, n_jobs=n_jobs,
                        n_genes=n_genes, getPval=getPval, verbose=verbose)
        
        #storing node attributes
        node_eff = summDF[['gene_name','eff']].set_index('gene_name').T.to_dict('records')
        nx.set_node_attributes(graph_gc, node_eff[0], name='eff')
        node_sens = summDF[['gene_name','sens']].set_index('gene_name').T.to_dict('records')
        nx.set_node_attributes(graph_gc, node_sens[0], name='sens')
        node_impact = summDF[['gene_name','impact']].set_index('gene_name').T.to_dict('records')
        nx.set_node_attributes(graph_gc, node_impact[0], name='impact')

        # summDF['smallest_eigenvec'] = vectors[:, 2]
        if verbose == True:
            print('DONE')

        # Clustering
        if cluster == True:
            clustered_mat, summDF = clusterPRS(X=norm_prs_matrix, G=graph_gc, df=summDF,
                                        Kclusters=Kclusters,verbose=verbose)      
        else:
            clustered_mat = pd.DataFrame(norm_prs_matrix, index = np.array(graph_gc.nodes), columns= np.array(graph_gc.nodes))

    else:
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        if verbose:
            print('Number of sub-graphs: ', len(subgraphs))

        # Get subgraphs with atleast 10 genes
        sg_mask = np.array([True if len(i.nodes) > 10 else False for i in subgraphs])
        subgraphs_f = [subgraphs[i] for i in np.where(sg_mask == True)[0]]
        if verbose:
            print('Number of viable sub-graphs: ', len(subgraphs_f))
        
        summDFs = {}
        graphs = {}
        PRSmats = {}

        for i,nam in enumerate(subgraphs_f):
            # Get subgraph
            graph_gc = subgraphs_f[i]
            print('Sub Graph shape: ', nx.adjacency_matrix(graph_gc).shape)

            # Getting Laplacian matrix
            if weighted == True:
                L = nx.laplacian_matrix(graph_gc, weight='Val').todense()
            else:
                L = nx.laplacian_matrix(graph_gc, weight=None).todense()

            # Eigen decomposition
            if verbose == True:
                print('Performing eigen decomposition...')
            n_genes = L.shape[0]
            if n_genes <= 10:
                    sys.exit("Too few genes")
        
            values, vectors = decompose(L=L, n_genes=n_genes, n_comps = n_comps,
                                        explainedV= explainedV,
                                        verbose=verbose)
            
            # PRS matric computation
            norm_prs_matrix = computePRS(values=values, n_genes=n_genes,
                                    vectors=vectors, verbose=verbose)
            
            # Get summary metrics
            summDF = getSummaryDF(X=norm_prs_matrix, G=graph_gc, L=L, alpha=alpha,
                            n_genes=n_genes, verbose=verbose)
            
            #storing node attributes
            node_eff = summDF[['gene_name','eff']].set_index('gene_name').T.to_dict('records')
            nx.set_node_attributes(graph_gc, node_eff[0], name='eff')
            node_sens = summDF[['gene_name','sens']].set_index('gene_name').T.to_dict('records')
            nx.set_node_attributes(graph_gc, node_sens[0], name='sens')
            node_impact = summDF[['gene_name','impact']].set_index('gene_name').T.to_dict('records')
            nx.set_node_attributes(graph_gc, node_impact[0], name='impact')

            #summDF['smallest_eigenvec'] = vectors[:, 2]
            if verbose == True:
                print('DONE')

            # Clustering
            if cluster == True:
                clustered_mat, summDF = clusterPRS(X=norm_prs_matrix, G=graph_gc, df=summDF,
                                            Kclusters=Kclusters,verbose=verbose)      
            else:
                clustered_mat = pd.DataFrame(norm_prs_matrix, index = np.array(graph_gc.nodes), columns= np.array(graph_gc.nodes))
            
            summDF['subgraph'] = i
            
            PRSmats[i] = clustered_mat
            graphs[i] = graph_gc
            summDFs[i] = summDF

        clustered_mat = PRSmats
        graph_gc = graphs
        DFlist = [df for key,df in summDFs.items()]
        summDF = pd.DataFrame()

        if len(DFlist) > 1:
            for df in DFlist:
                summDF = pd.concat([summDF, df], axis=0)
        else:
            summDF = summDFs[0]

    output = {'PRSmatrix': clustered_mat, 'Summary':summDF, 'Graph':graph_gc}
    return output


def get_graph_summary(G):
    df = pd.DataFrame()
    L = nx.laplacian_matrix(G, weight=None).todense()
    df['gene_name'] = list(G.nodes)
    df['deg'] = np.diag(L)
    closeness_centr = nx.closeness_centrality(G)
    df['closeness_centr'] = [closeness_centr[i] for i in closeness_centr]
    try:
        eigenvector_centr = nx.eigenvector_centrality_numpy(G)
        df['eigenvec_centr'] = [eigenvector_centr[i]
                                    for i in eigenvector_centr]
    except:
        print("Eigen vector centrality not computed")
        pass

    return df


