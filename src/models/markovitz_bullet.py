import numpy as np
from scipy import optimize

import plotly.express as px
import plotly.graph_objects as go
from numbers import Number
from copy import deepcopy

class MarkovitzSimulator:
    """
    Build a Markovitz Simulator

    Parameters
    ----------
    returns : np.ndarray
        a 2D array of returns
    simulations : tuple[np.ndarray], optional
        A tuple of 1D arrays (mu, sigma), by default None
    periods : int, optional
        The frequency of the returns. If yearly, 1, by default 1
    """

    def __init__(self,
                 returns : np.ndarray,
                 simulations: tuple[np.ndarray] = None,
                 sharpe_ratio: np.ndarray = None,
                 weights : np.ndarray = None, 
                 periods: int = 1) -> None:

        self.returns = returns
        self.simulations = simulations
        self.sharpe_ratio = sharpe_ratio
        self.weights = weights
        self.periods = periods

    @staticmethod
    def _generate_rand_weights(n : int) -> np.ndarray:
        """
        Generates n random weights between 0 and 1
        
        If n is a number, creates a 1D array. If n is a tuple, creates a 2D array 
        of weights. Each column is a portfolio and each row is the weight for a 
        given asset in the portfolio.  

        Parameters
        ----------
        n : int, tuple
            The number of required weights.
            Can be an int or tuple. If int returns a 1D array, if tuple returns a 
            2D array

        Returns
        -------
        np.ndarray
            An array of weights summing to 1 for each column. 
        """
        if isinstance(n, Number):
            n = tuple(n)

        k = np.random.rand(*n)

        return k / np.sum(k, axis = 0)

    def _update_attr(self, new_vals : list[tuple]) -> "MarkovitzSimulator":
        """
        Inside function that is used to reinstantiate a class with updated parameters
        
        Takes the class' parameters as a dict, updates it with new_vals and return
        an updated class by offloading the dictionnary in the class constructor.

        Parameters
        ----------
        new_vals : list[tuple]
            A list of tuples with each tuple being a pair (key, value)

        Returns
        -------
        MarkovitzSimulator
            A new instance of the updated class
        """

        attr = self.__dict__
        attr.update(new_vals)

        return MarkovitzSimulator(**attr)
    
    @staticmethod
    def _cov_rescaler(returns : np.ndarray, replace_value : np.ndarray, bias: bool = True) -> tuple[np.ndarray]:
        """
        Inside function used to rescale the covariance when used on values containing nan. 
        
        Since each value in a cov matrix is divided by n or (n - 1), it can be biased when there are nan 
        values. The rescaler creates a matrix N of the same size as the cov matrix with the goal of doing 
        rescaled_C = C * n / N
        
        For a cov matrix, passing replace_value = mean(returns) suppresses the nan in the cov matrix if 
        the data are not centred. If it is centred, passing 0 has the same effect. 

        Parameters
        ----------
        returns : np.ndarray 
            The array for which to impute missing data
        replace_value : np.ndarray/float
            The value to replace the nan. Can be an array or a float depending on the shape of returns
            If returns is 1D, must be a float. If returns is 2D, must be a 1D array (1 value for each col)
        bias : bool, by default True
            How the cov matrix was calculated. If True, assumed to be calculated with n. If False, assumed
            to be calculated with n - 1

        Returns
        -------
        tuple[np.ndarray]
            A tuple with the returns transformed with imputed values and a matrix of dim (A * A), the same size
            as a cov matrix. Each cell is the min number of valid data between 2 assets.
        """
        
        returns_imputed = returns.copy()
        
        nan_indices = np.isnan(returns_imputed)
        
        #Fancy indexing flattens the array temporarily so returns[nan_indices] creates a 
        #1D array of booleans. np.asarray(nan_indices).nonzero()[1] produces a 1D array of
        #the column index for the nan values. 
        #np.nanmean(axis = 0) creates a 1D array with length = max(np.asarray(nan_indices).nonzero()[1])
        #since it is the mean for each column. 
        #np.take(a, b) maps each index value in b to the corresponding value in a, matched by index.
        #e.g. a[b] for val in b
        #End result is each nan value replaced by the col mean, i.e. asset mean

        if len(np.squeeze(returns_imputed).shape) == 1:
            returns_imputed = np.where(nan_indices, replace_value, returns_imputed)
        else:
            returns_imputed[nan_indices] = np.take(replace_value, np.asarray(nan_indices).nonzero()[-1])

        nan_col_sums = np.sum(~nan_indices, axis=0)                
        n_mask = np.minimum.outer(nan_col_sums, nan_col_sums)
        n = len(returns)
        if not bias:
            n = n - 1 
        rescale_factor = n / n_mask
        
        return returns_imputed, rescale_factor

    def make_random_portfolios(self, n_simulations: int = 1) -> "MarkovitzSimulator":
        """
        Returns the mean and standard deviation of returns for a
        defined number of random portfolios

        Parameters
        ----------
        n_simulations : int, optional
            The number of random portfolios to generate, by default 1

        Returns
        -------
        MarkovitzSimulator
            An instance of the class with the simulated values
        """

        mean_returns = np.nanmean(self.returns, axis = 0)
                
        returns, rescale_factor = self._cov_rescaler(self.returns, mean_returns, bias = True)

        #We should be dividing by the number of actual existing values. So we rescale by the 
        # rescale matrix = n / n_mask to get the "right" cov value and ignoring the nan. If 
        # there is no nan, n / n_mask = 1

        covariance_matrix = np.cov(returns, rowvar = False, bias = True) * rescale_factor

        #If there is only one asset, weights are non-important. Result can be directly computed. 
        #Scaler is used to match the number of simulations required that will all be identical

        if len(np.squeeze(self.returns).shape) == 1:

            weights = np.ones(n_simulations)
            simulations = (weights * mean_returns, weights * covariance_matrix ** 0.5)
            updated_params = [("simulations", simulations), ("weights", weights)]
            
            return self._update_attr(updated_params)

        dim_weights = (returns.shape[1], n_simulations)
        weights = self._generate_rand_weights(dim_weights)

        mu = weights.T @ mean_returns 

        # We have W (A, N), W.T (N, A) and C (A, A)
        # So W.T @ C = R (N, A) and R @ W (N, N)
        # But for R @ W, only the product Ri,k @ Wk,i is meaningfull
        # for each set of weighs i since any other product would be
        # mixing up weights.Therefore we transpose R and do a outer
        # product before summing.This is a reworked dot product
        # calculated only for the elements that would create the
        # diagonal of the final matrix, saving us N * (N-1) computations

        #Slow version
        #sigma = (weights.T @ covariance_matrix @ weights)
        #sigma = np.diag(sigma) ** 0.5

        #Fast version
        sigma = weights.T @ covariance_matrix
        sigma = np.sum(sigma.T * weights, axis = 0) ** 0.5

        #Extract the class attributes, update with new attributes
        #and reinstantiate
        simulations = (mu, sigma)
        updated_params = [("simulations", simulations), ("weights", weights)]

        return self._update_attr(updated_params)

    def compute_sharpe_ratios(self, risk_free_rate: float = 0)-> "MarkovitzSimulator":
        """
        Computes sharpe ratios based on calculated mus and sigmas

        Parameters
        ----------
        risk_free_rate : float, optional
            The risk free rate, by default 0

        Returns
        -------
        MarkovitzSimulator
            An updated simulator
        """

        mu, sigma = self.simulations
        sharpe_ratio = ((mu - risk_free_rate) 
                        * self.periods ** 0.5
                        / sigma)
        
        updated_params = [("sharpe_ratio", sharpe_ratio)]
        return self._update_attr(updated_params)

    def plot(self, title: str = "Markovitz Bullet") -> go.Figure:
        """
        Plots a scatter plot of the mus against the sigmas
        
        If the instance contains sharpe ratios, each point is coloured
        with its sharpe ratio

        Parameters
        ----------
        title : str, optional
            The plot's title, by default "Markovitz Bullet"

        Returns
        -------
        go.Figure
            A plotly scatter plot
        """

        mu = self.simulations[0] * 100
        sigma = self.simulations[1] * 100

        if self.sharpe_ratio is not None:
            best_sharpe_index = np.argmax(self.sharpe_ratio)
            best_mu = mu[best_sharpe_index]
            best_sigma = sigma[best_sharpe_index]

            mu = np.delete(mu, best_sharpe_index)
            sigma = np.delete(sigma, best_sharpe_index)
            color = np.delete(self.sharpe_ratio, best_sharpe_index)
            color = np.round(color, 2)
            best_sharpe = np.max(color)

        else:
            color = ["blue"] * len(sigma)

        fig = px.scatter(x = sigma, y = mu, title = title, color = color, color_continuous_scale = "viridis")

        if self.sharpe_ratio is not None:
            fig.add_trace(go.Scatter(x = [best_sigma], 
                                     y = [best_mu],
                                     text = [best_sharpe],
                                     showlegend = False,  
                                     hovertemplate = "Sharpe : %{text}",
                                     marker_color = "red", 
                                     mode="markers",
                                     name = "Optimal Allocation",
                                     marker=dict(size=5,
                                                 symbol = "square"))
                                     )

        fig.update_layout(xaxis_title = "Sigma of Return",
                          yaxis_title = "Mean Return in %",
                          legend_title="Sharpe Ratio")

        return fig
