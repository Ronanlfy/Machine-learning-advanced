from scipy.spatial import distance
import numpy as np

# Call as create_index_set(evidence) where evidence
# is an n x m matrix describing the evidence for n data sets and m models.
# Pass the whole evidence matrix to the function rather than summing over the evidence for each data set.
#
# Script debugged and fixed by Arvid Fahlström Myrman.

def create_index_set(evidence):
    E = evidence.sum(axis=1)
    # change 'euclidean' to 'cityblock' for manhattan distance
    dist = distance.squareform(distance.pdist(evidence, 'euclidean'))
    np.fill_diagonal(dist, np.inf)
    
    L = []
    D = list(range(E.shape[0]))
    L.append(E.argmin())
    D.remove(L[-1])
    
    while len(D) > 0:
        # add d if dist from d to all other points in D
        # is larger than dist from d to L[-1]
        N = [d for d in D if dist[d, D].min() > dist[d, L[-1]]]
        
        if len(N) == 0:
            L.append(D[dist[L[-1],D].argmin()])
        else:
            L.append(N[dist[L[-1],N].argmax()])
        
        D.remove(L[-1])
    
    # reverse the resulting index array
    return np.array(L)[::-1]
