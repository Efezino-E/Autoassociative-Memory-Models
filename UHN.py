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
    
    def type(self):
        return self.similarity.name + " " + self.separator.name + " " + self.projection.name
        
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
        if self.data.shape[1] != input_data.shape[0]:
            raise ValueError("Input data and Stored memory not of the same dimensions")
        
        # Build similarity matrix
        similarity_matrix = []
        for point in self.data:
            diff = abs(point - input_data)
            similarity_matrix.append(1 / diff[np.logical_not(np.isnan(diff))].sum())
        
        similarity_matrix =  np.array(similarity_matrix)

        if np.inf in similarity_matrix:
            temp = np.array([0] * len(similarity_matrix))
            temp[np.argmax(similarity_matrix)] = 100
            similarity_matrix = temp
        
        if self.show_progress:
            print(f"{self.name} similarity flow complete")

        return similarity_matrix
    
    def type(self):
        return self.name

class euclidean():
    def __init__(self, show_progress: bool = False) -> None:
        self.name = "Euclidean"
        self.show_progress = show_progress
        self.data = None

    def fit(self, data_patterns):
        # store data patterns
        self.data = data_patterns
        if self.show_progress:
            print(f"data stored for {self.name} separator")
        
    def flow(self, input_data):

        # Raise value error for incompatible dimensions
        if self.data.shape[1] != input_data.shape[0]:
            raise ValueError("Input data and Stored memory not of the same dimensions")
        
        # Build similarity matrix
        similarity_matrix = []
        for point in self.data:
            diff = (point - input_data) ** 2
            similarity_matrix.append(1 / diff[np.logical_not(np.isnan(diff))].sum())
        
        similarity_matrix =  np.array(similarity_matrix)

        if np.inf in similarity_matrix:
            temp = np.array([0] * len(similarity_matrix))
            temp[np.argmax(similarity_matrix)] = 100
            similarity_matrix = temp
        
        if self.show_progress:
            print(f"{self.name} similarity flow complete")

        return similarity_matrix
    
    def type(self):
        return self.name

# Separator classes
class softmax():
    def __init__(self, show_progress: bool = False) -> None:
        self.name = "Softmax"
        self.show_progress = show_progress
        self.exponent = 2.71828

    def fit(self, data):
        return None

    def flow(self, similarity_matrix):
        # Apply the softmax function to the similarity matrix
        
        exponents = self.exponent ** similarity_matrix
        probabilities = ((exponents) / exponents.sum())
        
        if self.show_progress:
            print(f"{self.name} separator flow complete")
        
        return probabilities
    
    def type(self):
        return self.name
    
class max():
    def __init__(self, show_progress: bool = False) -> None:
        self.name = "Max"
        self.show_progress = show_progress

    def fit(self, data):
        return None

    def flow(self, similarity_matrix):
        # Apply the softmax function to the similarity matrix
        
        probabilities = np.array([0] * len(similarity_matrix))
        probabilities[np.argmax(similarity_matrix)] = 1
        
        if self.show_progress:
            print(f"{self.name} separator flow complete")
        
        return probabilities
    
    def type(self):
        return self.name
    
class polynomial_10():
    def __init__(self, show_progress: bool = False) -> None:
        self.name = "10th Order Polynomial"
        self.show_progress = show_progress

    def fit(self, data):
        return None

    def flow(self, similarity_matrix):
        # Apply the softmax function to the similarity matrix
        
        probabilities = (similarity_matrix ** 10)
        probabilities = probabilities / probabilities.sum()
        
        if self.show_progress:
            print(f"{self.name} separator flow complete")
        
        return probabilities
    
    def type(self):
        return self.name
    
class polynomial_5():
    def __init__(self, show_progress: bool = False) -> None:
        self.name = "5th Order Polynomial"
        self.show_progress = show_progress

    def fit(self, data):
        return None

    def flow(self, similarity_matrix):
        # Apply the softmax function to the similarity matrix
        
        probabilities = (similarity_matrix ** 5)
        probabilities = probabilities / probabilities.sum()
        
        if self.show_progress:
            print(f"{self.name} separator flow complete")
        
        return probabilities
    
    def type(self):
        return self.name
    
class polynomial_2():
    def __init__(self, show_progress: bool = False) -> None:
        self.name = "2nd Order Polynomial"
        self.show_progress = show_progress

    def fit(self, data):
        return None

    def flow(self, similarity_matrix):
        # Apply the softmax function to the similarity matrix
        
        probabilities = (similarity_matrix ** 2)
        probabilities = probabilities / probabilities.sum()
        
        if self.show_progress:
            print(f"{self.name} separator flow complete")
        
        return probabilities
    
    def type(self):
        return self.name


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
    
    def type(self):
        return self.name
