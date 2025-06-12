import numpy as np

class Collection:
    """This Implementation of VectorDB class is implementation of prototype A"""

    def __init__(self, top_k_results = 1, search_method = "cosine"):
        """
        Initiallize Vector Collection
        *Parameters*
        top_k_results (default:1): Top k results to return
        search_method (search_method:"cosine"): Which search method to use, Options: ["cosine","manhattan","euclidean"]
        """

        self.__collection = np.empty((0,0), dtype=np.float32)
        self.__top_k = top_k_results
        self.__search_method = search_method

    def add(self):
        """
        Adds a vector to the collection; Adds the indexed vector if collection is initiallized with 
        """


    def search(self):
        pass

    def get_info(self):
        """
        Returns Informations: Top_k, Search Method, Collection Length
        """

        print(f"Top K Result: {self.__top_k}\n Search Method: {self.__search_method}\n Collection Length: {self.__collection.shape[0]}")

    def ___call__(self):
        pass
