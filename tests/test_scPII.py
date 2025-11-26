from networkx import powerlaw_cluster_graph
from pandas import DataFrame

def test_scPRS_no_cluster():
    G = powerlaw_cluster_graph(25, 1, 0.6, seed=0)
    inputnet = DataFrame(nx.adjacency_matrix(G).todense())

    PRSout = scPRS(inputnet, cluster = False, getPval=False)

    assert len(PRSout) == 3
    assert isinstance(PRSout['PRSmatrix'], DataFrame)
    assert isinstance(PRSout['Summary'], DataFrame)
