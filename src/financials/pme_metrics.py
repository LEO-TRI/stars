import pandas as pd
import numpy as np 
import numpy_financial as npf

from .financial_utils import _series_to_array, _find_series_index

class PmeMetrics:
    """
    Compute and hold KSPME and Direct Alpha metrics

    Has functions to rescale cashflows 
    
    Parameters to instantiate
    ----------
    contributions : array_like, ideally pd.Series
        Series containing contribution values over time.
    distributions : array_like, ideally pd.Series
        Series containing distribution values over time.
    net_asset_value : array_like, ideally pd.Series
        Series containing net asset value values over time.
    index_fund : array_like, ideally pd.Series
        Series containing index fund values over time (e.g., CAC40).
    discount_rate : float, optional, default: 0.1
        The discount rate used in calculations.
    periods: int, optional, by default 4
        The periods used to annualize IRR, by default data is assumed to be quarterly
        
    Notes
    -------
    For increased comparability, make sure to select an index with reinvested dividends
    """

    def __init__(self, contributions : pd.Series, 
                distributions : pd.Series, 
                net_asset_value : pd.Series, 
                index_fund : pd.Series, 
                daily_data: pd.Series=None,
                discount_rate: float=0.,
                periods: int=4) -> None:
        """
        Initialize an instance of the PmeMetrics class.

        Returns
        -------
        None
            The function initializes the PmeMetrics instance with the provided data.

        Notes
        -----
        - The input Series should represent values over time, and their alignment is important for accurate calculations.
        - The net cashflows are computed as distributions minus contributions.
        - The last_net_asset_value is computed as the last value of net_asset_value discounted by the discount_rate.
        - The ks_pme and direct_alpha attributes are computed using the compute_ks_pme and compute_direct_alpha methods.
        """

        self.contributions = _series_to_array(contributions)
        self.distributions = _series_to_array(distributions)
        self.daily_data = _series_to_array(daily_data)
        self.net_cashflows = self.distributions - self.contributions
        self.discount_rate = discount_rate
        
        #Index fund as in CAC40, not dataframe index. Rescaled so the first period t = 1 is 1 and then each 
        #subsequent period is expressed as a ratio of the evolution in value
        index_fund = _series_to_array(index_fund)  
        self.index_fund = index_fund / index_fund[0]

        #Keeps in memory the original data index if at least one argument has an index, else takes a default index 
        self.time_index = _find_series_index(contributions, distributions, index_fund) 
        #* (1 - discount_rate)
        self.net_asset_value = _series_to_array(net_asset_value) 
        self.periods = periods

    #PME functions
    def _future_value_calculator(self, cashflows : np.ndarray, daily_data: np.ndarray = None) -> np.ndarray:
        """
        Calculates the future value of cashflows if those cashflows were invested in a public index. 
        
        Future values are calculated with the index values from instant t to final instant T for each cashflow at instant t.  

        Parameters
        ----------
        cashflow : np.ndarray
            A cashflow column. Can be distribution or contribution

        Returns
        -------
        np.ndarray
            The future value of cashflows

        Raises
        ------
        ValueError
            If len(cashflow) =/= len(index)
        """
                
        if daily_data is None: 
            if not (len(cashflows) == len(self.index_fund)):
                raise ValueError("cashflows and index must be the same length")
            index_T = self.index_fund[-1] #the index's value at the last period T
        else:
            pass
            #daily_data = _series_to_array(daily_data)
            #index_T = np.mean(daily_data[-30:]) #the monthly mean in the last period 
        
        rescaled_cashflows = cashflows * (index_T / self.index_fund)

        return rescaled_cashflows
    
    def _value_adjuster(self, distributions : np.ndarray, contributions : np.ndarray) -> tuple[np.ndarray]:
        """
        Convenience function to correct the initial calculation of the FVC and FVD by adding the last value of the last_net_asset_value

        Parameters
        ----------
        distributions : np.ndarray
            A column of cash distributions
        contributions : np.ndarray
            A column of cash contributions

        Returns
        -------
        distributions, contributions : np.ndarray
            Two arrays, one of distributions and contributions in a tuple
        """

        #Some contributions are negative. In this case, they are treated as as a reversion of funds and attributed to distributions
        distributions = distributions - np.where(contributions < 0, contributions, 0) 
        contributions[contributions < 0] = 0
        
        return (distributions, contributions)
    
    def _nav_adjuster(self, cashflow : np.ndarray, nav_array : np.ndarray) -> np.ndarray:
        """
        Adds the last period remainding NAV to the cashflows, with a potential discount
                
        Parameters
        ----------
        cashflow : np.ndarray
            A column of cashsflows of the form distributions - contributions
        nav_array : np.ndarray
            An array with the NAV values

        Returns
        -------
        cashflow : np.ndarray
            A cashflow array with the last value updated
        """

        cashflow[-1] = cashflow[-1] + nav_array[-1] * (1 - self.discount_rate)

        return cashflow

    def _calculate_ln_nav(self, cashflows : np.ndarray) -> list:
        """
        Calculates the LN-PME NAV. 

        Requires cashflows to be of the form contributions - distributions

        Parameters
        ----------
        cashflow : np.ndarray
            A column of cashsflows of the form distributions - contributions
            
        Returns
        -------
        nav_list : np.list
            A list of NAV for the synthetic benchmark fund
        """
        
        index_fund_growth = self.index_fund[1:] / self.index_fund[:-1]      

        #Each new NAV value at t for the PME is equal to the NAV at t - 1 times the change in the 
        #index + (-) the contributions (the distributions)

        periods = len(cashflows) - 1
        nav_list = [cashflows[0]]
        for i in range(periods):
            new_nav_val = nav_list[-1] * index_fund_growth[i] + cashflows[i + 1]
            nav_list.append(new_nav_val)

        return nav_list

    def compute_ks_pme(self, 
                       return_details: bool=False) -> float:
        """
        Calculates the Kaplan Schoar PME based on a passed index and the fund's cashflows*
        
        The formula is KSPME = FutureValuesDistributions / FutureValuesContributions
        
        Parameters
        ----------
        return_details : np.bool
            Whether to return the cumulated cashflows instead.
        
        Notes
        ----------
        With the cumulated cashflows, each value for t is equal to the sum of the capitalized cashflows
        So the ratio of the last cumulated capitalized distribution and contribution is the overall KS PME
            
        Returns
        -------
        ks_pme : float
            The KS-PME of the fund 
            
        or pd.DataFrame, the detailed cashflows, if return_details is True
        """
        
        #Computing cumulative capitalized cashflows --> Upper triangular matrix of format 
        cumulated_distributions, cumulated_contributions = self._compute_ks_pme_cashflows()
        cumulated_distributions, cumulated_contributions = self._value_adjuster(cumulated_distributions, cumulated_contributions)

        #Summing up all cashflows for each timestep t        
        cumulated_distributions = np.sum(cumulated_distributions, axis=0)
        cumulated_contributions = np.sum(cumulated_contributions, axis=0)
        
        #Adding the cashflows at T that are not capitalized to the last period
        cumulated_distributions[-1] = cumulated_distributions[-1] + self.distributions[-1]
        cumulated_contributions[-1] = cumulated_contributions[-1] + self.contributions[-1]
        
        #Adding the values at t=0
        cumulated_distributions = np.insert(cumulated_distributions, 0, self.distributions[0])
        cumulated_contributions = np.insert(cumulated_contributions, 0, self.contributions[0])

        #Adjusting and adding NAV to follow procedure as defined in other PME methods
        cumulated_distributions = self._nav_adjuster(cumulated_distributions, self.net_asset_value)

        if (return_details is True):
            
            return pd.DataFrame(data = {"cumulatedDistributions": cumulated_distributions,
                                        "cumulatedContributions": cumulated_contributions,
                                        "ksPme": cumulated_distributions / cumulated_contributions
                                        },
                                index = self.time_index
                                )
        
        kspme = cumulated_distributions[-1] / cumulated_contributions[-1]
        
        return kspme

    def compute_ks_pme_old(self) -> float:
        """
        Calculates the Kaplan Schoar PME based on a passed index and the fund's cashflows*
        
        The formula is KSPME = FutureValuesDistributions / FutureValuesContributions

        Returns
        -------
        ks_pme : float
            The KS-PME of the fund 
        """
        
        future_value_distribution = self._future_value_calculator(self.distributions)
        future_value_contribution = self._future_value_calculator(self.contributions)
        future_value_distribution, future_value_contribution = self._value_adjuster(future_value_distribution, future_value_contribution)
        future_value_distribution = self._nav_adjuster(future_value_distribution, self.net_asset_value)

        kspme = np.sum(future_value_distribution) / np.sum(future_value_contribution)
        
        return kspme
    
    def compute_direct_alpha(self) -> float:
        """
        Calculates the direct alpha based on a passed index and the fund's cashflows
        
        The formula is Alpha = IRR(FutureValuesContributions - FutureValuesDistributions)

        Returns
        -------
        direct_alpha : float
            The direct alpha of the fund
        """

        future_value_distribution = self._future_value_calculator(self.distributions)
        future_value_contribution = self._future_value_calculator(self.contributions)
        future_value_distribution, future_value_contribution = self._value_adjuster(future_value_distribution, future_value_contribution)
        future_value_distribution = self._nav_adjuster(future_value_distribution, self.net_asset_value)

        net_cashflows = future_value_distribution - future_value_contribution
        irr = npf.irr(net_cashflows) 

        return (1 + irr) ** (self.periods) - 1
    
    def compute_ln_pme(self) -> float:
        """
        Computes the Long Nickels PME metric
        
        All cashflows are the same until period T. At period T, the cashflow 
        is summed with a rescaled NAV_T.
        
        The NAV is built iteratively for period t based on values at t - 1 with 
        the formula NAV_t = NAV_t-1 * (Index_t / Index_t-1) + contribution_t - distributions_t 

        Returns
        -------
        float
            The LN PME metric expressed in % 
        """

        distributions, contributions = self._value_adjuster(self.distributions, self.contributions)
        net_cashflows = distributions - contributions

        #Only difference between pme_cashflows & pe_cashflow is the last NAV value
        #Add the last value of the PME NAV to the last cashflow

        pme_cashflows = -1 * net_cashflows.copy() #We change sign so that contributions are positive and distributions negative
        nav_list = self._calculate_ln_nav(pme_cashflows)

        pme_cashflows = net_cashflows.copy()
        pme_cashflows = self._nav_adjuster(pme_cashflows, nav_list)

        #Add the last value of the PE NAV to the last cashflow
        private_cashflows = net_cashflows.copy()
        private_cashflows = self._nav_adjuster(private_cashflows, self.net_asset_value)

        irr_pe = npf.irr(private_cashflows)
        irr_pe = (1 + irr_pe) ** self.periods - 1
        irr_pme = npf.irr(pme_cashflows)
        irr_pme = (1 + irr_pme) ** self.periods - 1

        return irr_pe - irr_pme
    
    def compute_mpme(self, 
                     return_details: bool=False) -> float:
        """
        Computes Cambridge's mPME metric

        Parameters
        ----------
        return_details : bool, optional
            Whether to return the rescaled cashflows or the PME metric, by default False
            If false, return the mPME metric

        Returns
        -------
        float
            The mPME metric
        """
        
        distributions, contributions = self._value_adjuster(self.distributions, self.contributions)
        distribution_weights = distributions / (distributions + self.net_asset_value)
        index_fund_growth = self.index_fund[1:] / self.index_fund[:-1]      
        
        periods = len(distribution_weights)
        nav_list = [contributions[0] - distributions[0]]
        distribution_list = [distributions[0]]
        
        for i in range(1, periods): #We need to rebuild the NAV iteratively
            
            value_creation = (nav_list[-1] * index_fund_growth[i - 1] + contributions[i])
        
            new_nav_val = value_creation * (1 - distribution_weights[i])        
            nav_list.append(new_nav_val)
            
            weighted_distribution = value_creation * distribution_weights[i]
            distribution_list.append(weighted_distribution)

        if (return_details is True):
            return pd.DataFrame(data = {"weightedDistributions": distribution_list,
                                        "distributions": distributions,
                                        "contributions": contributions,
                                        },
                                index = self.time_index
                                )
            
        net_cashflows_pme = np.asarray(distribution_list) - contributions
        pme_cashflows = self._nav_adjuster(net_cashflows_pme, nav_list)
        irr_pme = npf.irr(pme_cashflows)
        irr_pme = (1 + irr_pme) ** self.periods - 1
        
        net_cashflows = distributions - contributions
        private_cashflows = self._nav_adjuster(net_cashflows, self.net_asset_value)
        irr_pe = npf.irr(private_cashflows)
        irr_pe = (1 + irr_pe) ** self.periods - 1
        
        return irr_pe - irr_pme
    
    def _compute_ks_pme_cashflows(self,) -> tuple:
            """
            Computes a view of cumulated capitalized cashflows for a fund
            
            The cumulative cashflows are calculated as the sum of the capitalized cashflows for each time step t
            

            Returns
            -------
            pd.DataFrame
                A dataframe with 2 columns: 
            """
            
            index_fund_growth = self.index_fund[1:] / self.index_fund[:-1] #length is T - 1, we are missing t=0
            n_steps = len(index_fund_growth)
            #Since np.tri(n_steps).T is an upper triangular matrix filled with 1s and 0s otherwise, the product with the broadcast
            #is index_fund_growth in a n * n format but only the upper triangular part
            cumprod_matrix = ((index_fund_growth[None,:] + np.zeros(shape=(n_steps, n_steps), dtype=np.int8)) 
                              * np.tri(N=n_steps, dtype=np.int8).T
                              )           
            #Creates a square matrix that we will fill iteratively so that cashflows are only taken into account
            #once we reach the diagnonal value. Since the matrix is filled with 0, summing all rows means each cashflow gradual 
            #capitalization is only counted once the diagonal is passed going rightward
            cumprod_matrix[cumprod_matrix == 0] = 1 #NOTE find a better way
            cumprod_matrix = np.cumprod(cumprod_matrix, axis=1)
            cumprod_matrix =  np.triu(cumprod_matrix) #Keep only the upper triangle
            
            #Broadcasting values over the cumprod matrix. 
            cumulated_distributions = self.distributions[:-1,None] * cumprod_matrix 
            cumulated_contributions = self.contributions[:-1,None] * cumprod_matrix
            
            return (cumulated_distributions, cumulated_contributions)
                
    def compute_market_performances(self) -> tuple[float, float]:
        """
        Use the fund performance and the PME to calculate the corresponding benchmark performance

        Roughly removing market perf from fund perf gives the PME
        Removing the PME from the fund perf gives the market perf
        
        """
        distributions, contributions = self._value_adjuster(self.distributions, self.contributions)
        distributions = self._nav_adjuster(distributions, self.net_asset_value)

        #DIRECT ALPHA
        irr_base = (1 + npf.irr(distributions - contributions)) ** self.periods - 1
        market_irr = irr_base - self.direct_alpha()

        #KSPME
        tvpi = distributions / contributions
        market_multiple = tvpi / self.compute_ks_pme()

        return (market_irr, market_multiple)


