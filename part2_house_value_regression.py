import torch
import pickle
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelBinarizer

class Regressor():

    def __init__(self, x, nb_epoch = 1000, hidden_size = 0, activation_function = "identity", learning_rate = 0.01):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.hidden_size = hidden_size
        self.activation_function = activation_function

        if hidden_size > 0:
            if self.activation_function == "identity":
                self.model = torch.nn.Sequential (
                    torch.nn.Linear(self.input_size, self.hidden_size),
                    torch.nn.Linear(self.hidden_size, self.output_size)
                )
            elif self.activation_function == "relu":
                self.model = torch.nn.Sequential (
                    torch.nn.Linear(self.input_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.output_size)
                )
            elif self.activation_function == "sigmoid":
                self.model = torch.nn.Sequential (
                    torch.nn.Linear(self.input_size, self.hidden_size),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(self.hidden_size, self.output_size)
                )
        else:
            self.model = torch.nn.Linear(self.input_size, self.output_size)
        self.criterion = torch.nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Fill missing values in x with median (quant) or mode (qual)
        if training:
            self.vals = x.loc[:, x.columns != "ocean_proximity"].median().to_dict()
            self.vals["ocean_proximity"] = x.loc[:, x.columns == "ocean_proximity"].mode().to_dict()["ocean_proximity"][0]
        x = x.fillna(self.vals)
        # If training initialise label binarizer for ocean proximity
        if training:
            self.encoder = LabelBinarizer()
            self.encoder.fit(x['ocean_proximity'])
        
        # Apply label binarization to x
        transformed = self.encoder.transform(x['ocean_proximity'])
        ohe_df = pd.DataFrame(transformed)
        x = pd.concat([x, ohe_df], axis=1).drop(['ocean_proximity'], axis=1)
        # Convert dataframes to numpy ndarrays
        x = x.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        # If training extract minmax values
        if training:
            self.x_max = x.max(axis=0)
            self.x_min = x.min(axis=0)
            if isinstance(y, np.ndarray):
                self.y_max = y.max(axis=0)
                self.y_min = y.min(axis=0)

        # Normalise values using MinMax normalisation
        x = (x - self.x_min)/(self.x_max-self.x_min)
        # Currently normalising y as this should help during training - remember to convert back!
        if isinstance(y, np.ndarray):
            y = (y - self.y_min)/(self.y_max-self.y_min)

        # Return preprocessed x and y, return None for y if it was None
        return x, (y if isinstance(y, np.ndarray) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        
    def fit(self, x, y, v_x=None, v_y=None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        early_stop = False
        if isinstance(v_x, pd.DataFrame) and isinstance(v_y, pd.DataFrame):
            early_stop = True
            stop_count = 1

        batch_size = 64
        batches = math.ceil(len(X)/batch_size)

        best_score = float('inf')
        old_score = float('inf')

        for epoch in range(self.nb_epoch):
            # Shuffle the data
            indices = np.random.permutation(len(X))
            X = X[indices]
            Y = Y[indices]

            # Perform mini-batch gradient descent
            for i in range(batches):
                start = i*batch_size
                if i < batches - 1:
                    end = (i+1)*batch_size
                else:
                    end = len(X)

                x_train_tensor = torch.from_numpy(X[start:end]).float()
                y_train_tensor = torch.from_numpy(Y[start:end]).float()

                x_train_tensor.requires_grad_(True)
                self.optimiser.zero_grad()
                y_hat = self.model(x_train_tensor)
                loss = self.criterion(y_hat,y_train_tensor)
                loss.backward()
                self.optimiser.step()

            # Check early stop
            if early_stop:
                rmse = self.score(v_x,v_y)
                #print("rmse: {}n".format(rmse))
                if rmse < best_score:
                    best_score = rmse
                    best_model = self
                if stop_count == 10:
                    if rmse >= old_score:
                        return best_model
                    else:
                        stop_count = 0
                        old_score = rmse
                else:
                    stop_count += 1
                
            # print(f"Epoch: {epoch}\t w: {self.linear.weight.data[0]}\t b: {self.linear.bias.data[0]:.4f} \t L: {loss:.4f}")
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        with torch.no_grad():
            x_test = torch.from_numpy(X).float()
            p = self.model(x_test).numpy()
        #rescale predictions to convert back to standard form
        p = p*(self.y_max-self.y_min) + self.y_min
        return p

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        predictions = self.predict(x)
        y = y.to_numpy()
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        return rmse

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(data): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    rmse_best = float('inf')
    model_best_neurons = 0

    # Split data in to training, validation and test sets
    data.sample(frac=1).reset_index(drop=True)
    output_label = "median_house_value"
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    split_train = int(0.8 * len(x))
    split_val = int(0.9*len(x))
    x_train = x[:split_train].reset_index(drop=True)
    x_val = x[split_train:split_val].reset_index(drop=True)
    x_test = x[split_val:].reset_index(drop=True)
    y_train = y[:split_train].reset_index(drop=True)
    y_val = y[split_train:split_val].reset_index(drop=True)
    y_test = y[split_val:].reset_index(drop=True)

    learning_rates = [0.0001,0.001,0.01,0.1]
    activation_functions = ["identity", "relu", "sigmoid"]

    for neurons in range(0,14):
        for learning_rate in learning_rates:
            for activation_function in activation_functions:
                print("Training with: " + activation_function + ", " + str(neurons) + ", " + str(learning_rate))
                regressor = Regressor(x_train, nb_epoch=1000, activation_function=activation_function, hidden_size=neurons, learning_rate=learning_rate)
                regressor.fit(x_train, y_train, x_val, y_val)
                rmse_temp = regressor.score(x_val, y_val)
                print("rmse: {}\n".format(rmse_temp))
                if rmse_temp < rmse_best:
                    rmse_best = rmse_temp
                    model_best = regressor
                    model_best_neurons = neurons
                    model_best_activation = activation_function
                    model_best_learning = learning_rate
    
    rmse_test = model_best.score(x_test, y_test)
    save_regressor(model_best)
    
    print("Best model activation function: " + model_best_activation)
    print("Best model number of neurons: {}\n".format(model_best_neurons))
    print("Best model learning rate: {}\n".format(model_best_learning))
    print("\nBest regressor error: {}\n".format(rmse_test))
    
    return  model_best_neurons, model_best_activation, model_best_learning
    
    # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    data = pd.read_csv("housing.csv") 
    RegressorHyperParameterSearch(data)
    #example_main()

