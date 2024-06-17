import numpy as np
import faiss
import time
import torch
import torch.nn.functional as F
from torch_cluster import knn

def faissKNeighbors(x, k):
    index = None
    k = k

    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x.astype(np.float32))
    distances, indices = index.search(x.astype(np.float32), k=k)
    return distances, indices

def torchknn(xb, xq, k):
    bs_scores = F.linear(xq, xb, bias=None)  
    return bs_scores.topk(k=k, dim=1, sorted=True, largest=False)

def main():
    x = np.random.rand(100000, 2)
    k = 5
    start_time = time.time()
    print(faissKNeighbors(x, k))
    print("--- %s seconds ---" % (time.time() - start_time))
    x = torch.tensor(x).float()
    start_time = time.time()
    # print(torchknn(x, x, k))
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    print(knn(x, x, k))
    print("--- %s seconds ---" % (time.time() - start_time))

    
if __name__ == '__main__':
    main()