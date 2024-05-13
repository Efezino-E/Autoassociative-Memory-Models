import numpy as np

# === UHN MODEL CLASS 

class UHN_model:
    
    def __init__(self, similarity, separator, projection):
        """
        Creates a Unified Hopfield Model with a predefined similarity, separator and projection function
        """
        # instantiate model with the similarity, spearator and projector functions
        self.similarity = similarity
        self.separator = separator
        self.projection = projection

    def fit(self, input_data, output_data): # might be unecessary to the package at this moment
        """
        Stores or trains the Unified Hopfiled model based on the input and output data pair.
        """
        # convert data type to numpy arrays
        input_data = np.array(input_data)
        output_data = np.array(output_data)

        # raise error if input - output data associations are not complete
        if input_data.shape[0] != output_data.shape[0]:
            raise ValueError("Input and Output data not of the same dimensions. There are inputs not associated with outputs or vice versa")

        # store the data
        self.input_data = input_data
        self.output_data = output_data

    def predict(self, input):
        """
        Returns data that the Unified Hopfield Network associates with the input data.
        """
        # convert the input to a numpy array
        input = np.array(input)

        # first: calculate the similarity of the input to each input data stored
        num_patterns = self.input_data.shape[0]
        output = [0] * num_patterns

        for i in range(num_patterns):
            output[i] = self.similarity(input, self.input_data[i])
        
        output = np.array(output)

        # second: apply the separator function to magnify differences in the similarity vector
        output = self.separator(output)

        # third: apply the projection function to create a final output based on the input output association
        output = self.projection(output, self.output_data)

        return output

    def structure(self):
        return (self.similarity, self.separator, self.projection)

# === PREDEFINED SIMILARITY FUNCTIONS ===

def manhattan(vector1, vector2):
    """
    calculates the multiplicative inverse of the manhattan distance between two vectors
    """
    distance = np.sum(np.abs(vector1 - vector2))

    if distance == 0:
        return np.inf
    else:
        return 1 / distance
    
def euclidean(vector1, vector2):
    """
    calculates the multiplicative inverse of the euclidean distance between two vectors
    """
    distance = np.sum((vector1 - vector2) ** 2)

    if distance == 0:
        return np.inf
    else:
        return 1 / distance
    
def cosineSimilarity(vector1, vector2):
    """
    calculates the cosine similarity between two vectors
    """
    distance = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

    if distance == 0:
        return np.inf
    else:
        return 1 / distance

# === PREDEFINED SEPRATOR FUNCTIONS ===

def softMax(vector):
    """
    Exponentially magnifies the difference between components of the input vector.
    Further scales the vector such that it's components sum up to 1
    """
    # If one or more components are infinite, sclae the first infinite component to 1 and the rest to zero
    for i in range(len(vector)):
        if vector[i] == np.inf:
            vector = [0] * len(vector)
            vector[i] = 1
            return np.array(vector)
        
    vector = np.exp(vector)
    vector = vector / np.sum(vector)
    return vector

def Max(vector):
    """
    Finds the maximum value of the vector's component
    changes the first occurence of this vector to 1
    and the other components to zero
    """
    max_index = vector.argmax()
    vector = np.zeros(len(vector))
    vector[max_index] = 1
    return vector

def nthPolynomial(n):
    """
    Magnifies the differences between components of a vector using a polynomial function of order n.
    Further scales the vector such that it's components sum up to 1
    """
    def polynomial(vector, exponent = n):
        # If one or more components are infinite, sclae the first infinite component to 1 and the rest to zero
        for i in range(len(vector)):
            if vector[i] == np.inf:
                vector = [0] * len(vector)
                vector[i] = 1
                return np.array(vector)
        vector = vector ** exponent
        vector = vector / np.sum(vector)
        return vector

    return polynomial


# === PREDEFINED PROJECTION FUNCTIONS ===

def weightedSum(weights, output_data):
    """
    Calculates a weighted sum of different outputs contained within the output data. the 
    """
    values = np.transpose(output_data)
    dimension = values.shape[0]
    output = [0] * dimension

    for i in range(dimension):
        output[i] = np.dot(weights, values[i])
    
    return output
