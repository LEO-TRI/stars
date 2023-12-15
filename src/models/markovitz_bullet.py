import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from numbers import Number

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
                 periods: int = 1) -> None:

        self.returns = returns
        self.simulations = simulations
        self.sharpe_ratio = sharpe_ratio
        self.periods = periods

    @staticmethod
    def _generate_rand_weights(n : int) -> np.ndarray:
        """
        Generates n random weights between 0 and 1

        Parameters
        ----------
        n : int, tuple
            The number of required weights.
            Can be an int or tuple

        Returns
        -------
        np.ndarray
            An array of weight summing to 1
        """
        if isinstance(n, Number):
            n = tuple(n)

        k = np.random.rand(*n)

        return k / np.sum(k, axis = 0)

    def _update_attr(self, key : str, value) -> "MarkovitzSimulator":

        attr = self.__dict__
        attr[key] = value

        return MarkovitzSimulator(**attr)

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

        mean_returns = np.nanmean(self.returns, axis=0) * self.periods
        covariance_matrix = np.cov(self.returns, rowvar=False) * self.periods

        #If there is only one asset, weights are non-important
        #Result can be directly computed. Scaler is used to match
        #the number of simulations required that will all be identical
        if len(np.squeeze(self.returns).shape) == 1:
            scaler = np.zeros(n_simulations)
            simulations = (scaler + mean_returns, scaler + covariance_matrix)
            return MarkovitzSimulator(self.returns, simulations)

        dim_weights = (self.returns.shape[1], n_simulations)
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

        simulations = (mu, sigma)

        #Extract the class attributes, update with new attributes
        # and reinstantiate

        return self._update_attr("simulations", simulations)

    def compute_sharpe_ratios(self, risk_free_rate: int = 0):

        mu, sigma = self.simulations
        sharpe_ratio = (mu - risk_free_rate) / sigma

        return self._update_attr("sharpe_ratio", sharpe_ratio)

    def plot(self, title: str = "Markovitz Bullet"):

        mu = self.simulations[0] * 100
        sigma = self.simulations[1] * 100

        if self.sharpe_ratio is not None:
            color = self.sharpe_ratio
            best_mu = mu[np.argmax(self.sharpe_ratio)]
            best_sigma = sigma[np.argmax(self.sharpe_ratio)]
        else:
            color = ["blue"] * len(sigma)

        fig = px.scatter(x = sigma, y = mu, title = title, color = color, color_continuous_scale = "viridis")

        if self.sharpe_ratio is not None:
            fig.add_trace(go.Scatter(x = [best_sigma], y = [best_mu], marker_color = "red", mode="markers"))

        fig.update_layout(xaxis_title = "Sigma of Return",
                          yaxis_title = "Mean Return in %")

        return fig
