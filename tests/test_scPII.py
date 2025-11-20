import networkx as nx
import pandas as pd
from scPII.core import scPRS

def test_scPRS_no_cluster():
    G = nx.powerlaw_cluster_graph(25, 1, 0.6, seed=0)
    inputnet = pd.DataFrame(nx.adjacency_matrix(G).todense())

    PRSout = scPRS(inputnet, cluster = False, getPval=False)

    assert len(PRSout) == 3
    assert isinstance(PRSout['PRSmatrix'], pd.DataFrame)
    assert isinstance(PRSout['Summary'], pd.DataFrame)
