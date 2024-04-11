# Work in progress
# --- to be better structured as an Object oriented class
# --- to involve more similarity, separator and projection functions
# --- to implement iterators

import numpy as np

class UHNModel:
    
    def __init__(self, similarity = None, separator = None, projection = None, iterator = None, quiet = True) -> None:
        # Instantiate with the similarity, separator, projector and iterator classes to use for the UHN model
        self.similarity = similarity(show_progress = not quiet)
        self.separator = separator(show_progress = not quiet)
        self.projection = projection(show_progress = not quiet)
        self.iterator = iterator
    
    def fit(self, data):
        # Fit the similarity, separator and projection functions based on the data
        self.similarity.fit(data)
        self.separator.fit(data)
        self.projection.fit(data)
    
    def query(self, input_data):

        if not self.iterator:
            # flow the input through the framework's pipeline
            out1 = self.similarity.flow(input_data)
            out2 = self.separator.flow(out1)
            out3 = self.projection.flow(out2)
            return out3 
        
        else:
            # Not yet created iterator types
            raise ValueError("Iterator types not yet defined")
        
# Similarity classes
class manhattan():
    def __init__(self, show_progress: bool = False) -> None:
        self.name = "Manhattan"
        self.show_progress = show_progress
        self.data = None

    def fit(self, data_patterns):
        # store data patterns
        self.data = data_patterns
        if self.show_progress:
            print(f"data stored for {self.name} separator")
        
    def flow(self, input_data):

        # Raise value error for incompatible dimensions
        if self.data.shape[0] != input_data.shape[0]:
            raise ValueError("Input data and Stored memory not of the same dimensions")
        
        # Build similarity matrix
        similarity_matrix = []
        for point in self.data:
            similarity_matrix.append(abs(point - input_data).sum())
        
        similarity_matrix =  np.array(similarity_matrix)
        maxi = max(similarity_matrix)
        similarity_matrix = maxi - similarity_matrix
        
        if self.show_progress:
            print(f"{self.name} similarity flow complete")

        return similarity_matrix

# Separator classes
class softmax():
    
    def __init__(self, show_progress: bool = False, invert_probabilities: bool = False) -> None:
        self.name = "Softmax"
        self.show_progress = show_progress
        self.exponent = 2.71828
        self.invert_probabilities = invert_probabilities

    def fit(self, data):
        return None

    def flow(self, similarity_matrix):
        # Apply the softmax function to the similarity matrix
        
        exponents = self.exponent ** similarity_matrix
        probabilities = ((exponents) / exponents.sum())
        
        if self.show_progress:
            print(f"{self.name} separator flow complete")
        
        return probabilities


# Projection classes
class weighted_sum():
    
    def __init__(self, show_progress: bool = False) -> None:
        self.name = "Weighted Sum"
        self.show_progress = show_progress
    
    def fit(self, data_patterns):
        self.data = np.transpose(data_patterns)
        if self.show_progress:
            print(f"data stored for {self.name} projection")
    
    def flow(self, similarity_matrix):
        output = []

        for val in range(self.data.shape[0]):
            output.append(np.dot(similarity_matrix, self.data[val]))
        
        if self.show_progress:
            print(f"{self.name} projection applied")
            
        return output
