import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances

class Collection:
    """This Implementation of VectorDB class is implementation of prototype A"""

    def __init__(self, 
                 embedding_length = 768, 
                 embedding_dim = np.float32,
                 top_k_results = 1, 
                 search_method = "cosine"):
        """
        Initiallize Vector Collection
        
        *parameters*
        top_k_results (default:1): Top k results to return
        
        search_method (search_method:"cosine"): Which search method to use, Options: ["cosine","manhattan","euclidean"]
        """

        self.__collection = np.empty((0,embedding_length), dtype=embedding_dim)
        self.__embedding_length = embedding_length
        self.__top_k = top_k_results
        self.__search_method = search_method

    def add(self, vector:np.ndarray):
        """
        Adds a vector to the collection; Adds the indexed vector if collection is initiallized with 
        
        *parameters*

        vector: <np.array> Vector to add in the collection
        """
        
        assert vector.ndim == 2 and vector.shape[1] == self.__embedding_length, f"Vector must be a 2D array and Equal to {self.__embedding_length}"
        
        if self.collection.shape[0] == 0:
            self.collection = vector.reshape(1, -1)
        else:
            self.collection = np.vstack((self.collection, vector.reshape(1, -1)))


    def search(self, query:np.ndarray):
        assert query.ndim == 2 and query.shape[1] == self.__embedding_length, f"Vector must be a 2D array and Equal to {self.__embedding_length}"

        match self.__search_method: # TODO: Include More search method
            case "cosine":
                return cosine_similarity(self.collection, query).flatten().argsort()[:self.__top_k]
            case "euclidean":
                return euclidean_distances(self.collection, query).flatten().argsort()[:self.__top_k]
            case "manhattan":
                return manhattan_distances(self.collection, query).flatten().argsort()[:self.__top_k]
            case _:
                raise ValueError("Unsupported search method. Use 'cosine', 'euclidean', or 'manhattan'.")
                return None
    

    def get_info(self):
        """
        Returns Informations: Top_k, Search Method, Collection Length
        """

        print(f"Top K Result: {self.__top_k}\n Search Method: {self.__search_method}\n")
        print(f"Collection Length: {self.__collection.shape[0]}\n Embedding Length: {self.__embedding_length}\n")
        print(f"Collection Type: {self.__collection.dtype}")

    def ___call__(self, query:np.ndarray):
        return self.search(query)
