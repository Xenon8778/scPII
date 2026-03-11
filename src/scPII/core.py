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
            if n_comps < 5:
                sys.exit("Error: Less than 5 Eigenpairs explain {explainedV} variance. Please increase 'explainedV'.")
        
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

def shuffle_PRSmatrix(PRSmatrix, seed=1):
    rng = np.random.default_rng(seed)
    idxrow = rng.permutation(PRSmatrix.shape[0])
    idxcol = rng.permutation(PRSmatrix.shape[0])
    return PRSmatrix[np.ix_(idxrow, idxcol)]

def shuffled_impacts(PRSmatrix, niter=1000, alpha=1):
    mat = PRSmatrix.values
    genes = PRSmatrix.index
    n = mat.shape[0]

    # weights matrix
    W = 1 - np.eye(n)

    # preallocate result matrix
    res = np.zeros((n, niter))

    for i in tqdm(range(niter)):
        shuffledMat = shuffle_PRSmatrix(mat, seed=i)

        eff = np.average(shuffledMat, weights=W, axis=1)
        sens = np.average(shuffledMat, weights=W, axis=0)

        eff = np.log1p(eff)
        sens = np.log1p(sens) * alpha
        impact = sens - eff

        # fast min-max scaling
        impact = (impact - impact.min()) / (impact.max() - impact.min())
        res[:, i] = impact

    resDF = pd.DataFrame(res, index=genes,
                         columns=[f'impact_{i}' for i in range(niter)])
    
    return resDF

def getPvals(X,
            df,
            niter: int=10000,
            alpha: float=1.0,
            two_sided=False,
            verbose: bool=True):
    """
    Compute empirical p-values comparing observed impact
    against shuffled PRS impact distributions.

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
    two_sided : bool
        Whether to compute two-sided p-values    
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

    obs = df['impact'].values
    if verbose:
        print(f'Shuffling PRS matrix over {niter} iterations...')
    null = shuffled_impacts(X, niter=niter, alpha=alpha).values
    n_perm = null.shape[1]

    # Computes emperical Pvalues by shuffling the PRS matrix following Davison and Hinkley
    if verbose:
        print(f'Computing P-values...')
    if two_sided:
        pvals = (1 + np.sum(np.abs(null) >= np.abs(obs[:, None]), axis=1)) / (n_perm + 1)
    else:
        pvals = (1 + np.sum(null >= obs[:, None], axis=1)) / (n_perm + 1)

    # Optional Z-score
    # zscores = (obs - null.mean(axis=1)) / null.std(axis=1)

    res = pd.DataFrame({
        "impact_obs": obs,
        # "null_mean": null.mean(axis=1),
        # "null_sd": null.std(axis=1),
        # "zscore": zscores,
        "pval": pvals
    })

    P = res['pval']
    return P

def getSummaryDF(X, G, L, 
                getPval: bool = True,
                n_genes: int = None, 
                alpha: float = 1, 
                verbose: bool = True,
                n_boot: int = 10000):
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
        df_['pval'] = getPvals(X=pd.DataFrame(X, index=df['gene_name']),
                               df=df_, alpha=alpha, 
                               niter=n_boot,  
                               verbose=verbose)
    
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
        weighted: bool = True,
        explainedV: float = 0.1,
        n_boot: int = 1000,
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
    # Calculating giant component - Largest connected componenet of the graph
    if verbose == True:
        print('Calculating giant component...')
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
    
    # PRS matric computation
    norm_prs_matrix = computePRS(values=values, n_genes=n_genes,
                            vectors=vectors, verbose=verbose)
    
    # Get summary metrics
    summDF = getSummaryDF(X=norm_prs_matrix, G=graph_gc, L=L, alpha=alpha, n_boot=n_boot,
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


def differentialPRS(PRS_case,
                    PRS_control,
                    impact_threshold: float = 0.75,
                    alpha: float = 1.0,
                    eps: float = 0.1,
                    niter: int = 1000,
                    two_sided: bool = False,
                    verbose: bool = True):
    """
    Compute differential PRS impact between two conditions.

    Parameters
    ----------
    PRS_case : dict
        Output of scPRS() for case condition.
    PRS_control : dict
        Output of scPRS() for control condition.
    alpha : float, default=1
        Scaling factor used for sensitivity.
    eps : float, default=0.1
        Stabilizer for denominator in delta impact.
    niter : int, default=1000
        Number of permutations for null distribution.
    two_sided : bool, default=False
        Whether to compute two-sided empirical p-values.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    pd.DataFrame
        Differential impact results with empirical p-values.
    """

    if verbose:
        print("Computing differential impact...")

    if (set(PRS_case['Summary']['gene_name'] ) != set(PRS_control['Summary']['gene_name'])):
        sys.exit("PRS outputs don't have same genes. Ensure scPII has been run on GRNs with same genes.")        

    # Extract impact score
    impact_case = PRS_case['Summary'].set_index('gene_name')['impact']
    impact_ctrl = PRS_control['Summary'].set_index('gene_name')['impact']

    df = pd.DataFrame({
        "impactCase": impact_case,
        "impactControl": impact_ctrl
    })

    df = df.loc[impact_case.index.intersection(impact_ctrl.index)]

    # Differential impact
    df["delta_impact"] = (df["impactCase"] - df["impactControl"]) / (df["impactControl"] + eps)

    if verbose:
        print(f"Generating shuffled null distributions ({niter} iterations)...")

    # Permutation test
    shuffled_ctrl = shuffled_impacts(
        PRS_control['PRSmatrix'],
        niter=niter,
        alpha=alpha
    )

    shuffled_case = shuffled_impacts(
        PRS_case['PRSmatrix'],
        niter=niter,
        alpha=alpha
    )

    shuffledDelta = (shuffled_case - shuffled_ctrl) / (shuffled_ctrl + eps)

    obs = df["delta_impact"].values
    null = shuffledDelta.values

    n_perm = null.shape[1]

    if verbose:
        print("Computing empirical p-values...")

    if two_sided:
        pvals = (1 + np.sum(np.abs(null) >= np.abs(obs[:, None]), axis=1)) / (n_perm + 1)
    else:
        pvals = (1 + np.sum(null >= obs[:, None], axis=1)) / (n_perm + 1)

    df["pval"] = pvals

    if verbose:
        print("DONE")

    # Filter to return high impact genes only
    return df[df['impactCase'] > impact_threshold].sort_values("delta_impact", ascending=False)