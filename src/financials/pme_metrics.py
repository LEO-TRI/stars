import pandas as pd
import numpy as np 
import numpy_financial as npf
from functools import reduce

from .financial_utils import _series_to_array, _find_series_index
from .financial_graphs import _create_plotly_table

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
    """

    def __init__(self, contributions: pd.Series, 
               distributions: pd.Series, 
               net_asset_value: pd.Series, 
               index_fund: pd.Series, 
               discount_rate: float=0.1) -> None:
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
        self.net_cashflows = self.distributions - self.contributions
        self.discount_rate = discount_rate
        
        #Index fund as in CAC40, not dataframe index
        self.index_fund = _series_to_array(index_fund)  
        #Keeps in memory the original data index if at least one argument has an index, else takes a default index 
        self.time_index = _find_series_index(contributions, distributions, index_fund) 
        #* (1 - discount_rate)
        self.net_asset_value = _series_to_array(net_asset_value) 

    #PME functions
    def future_value_calculator(self, cashflow: np.ndarray) -> np.ndarray:
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

        if not (len(cashflow) == len(self.index_fund)):
            raise ValueError("cashflow and index must be the same length")

        index_T = self.index_fund[-1] #the index's value at the last period T

        IT_vec = np.ones(len(self.index_fund)) * index_T

        return cashflow * (IT_vec / self.index_fund)
    
    def future_value_adjuster(self) -> np.ndarray:
        """
        Convenience function to correct the initial calculation of the FVC and FVD by adding the last value of the last_net_asset_value

        Returns
        -------
        FVD, FVC : np.ndarray
            Two arrays, one of future distributions and one of future contributions
        """

        future_value_contribution = self.future_value_calculator(self.contributions)
        future_value_distribution = self.future_value_calculator(self.distributions)
        future_value_distribution[-1] = future_value_distribution[-1] + self.net_asset_value[-1] * (1 - self.discount_rate)

        #Some contributions are negative. In this case, they are treated as as a reversion of funds and attributed to distributions
        future_value_distribution = future_value_distribution + np.where(future_value_contribution<0, - 1 * future_value_contribution, 0) 
        future_value_contribution = np.where(future_value_contribution<0, 0, future_value_contribution)

        return future_value_distribution, future_value_contribution
    
    def compute_ks_pme(self) -> float:
        """
        Calculates the Kaplan Schoar PME based on a passed index and the fund's cashflows

        Returns
        -------
        ks_pme : float
            The KS-PME of the fund 

        Notes
        -------
        For increased comparability, make sure to select an index with reinvested dividends
        """

        future_value_distribution, future_value_contribution = self.future_value_adjuster()

        return np.sum(future_value_distribution)/np.sum(future_value_contribution)

    def compute_direct_alpha(self) -> float:
        """
        Calculates the direct alpha based on a passed index and the fund's cashflows

        Returns
        -------
        direct_alpha : float
            The direct alpha of the fund
        """

        future_value_distribution, future_value_contribution = self.future_value_adjuster()
        
        net_cashflows = future_value_distribution - future_value_contribution

        return npf.irr(net_cashflows)

class PmeTable(PmeMetrics):
    """
    Class to calculate and hold a table of PME metrics. 

    Should not be instantiated directly. Instead, use class constructors from_wide_data and 
    from_long_data.  
    """
    def __init__(self, table_data : np.ndarray, index_list : list, portfolio_list : list):

        self.index_list = index_list
        self.portfolio_list = portfolio_list
        
        self.table_data = table_data

        #Here as a reminder that self.table will be filled later 
        self.table = None 

    @classmethod
    def from_long_data(cls, 
                        pe_fund: pd.DataFrame, 
                        index_list: list, 
                        portfolio_list: list, 
                        contribution_col: str, 
                        distribution_col: str, 
                        fmv_col: str = "fmv", 
                        entity_col: str = "entity", 
                        metric: str = "kspme",
                        discount_rate: float = 0.2) -> "PmeTable":
        """
        Class constructor for PmeTable

        Generate a Public Market Equivalent (PME) table
        
        For the time being, requires a DataFrame in long format. If you have wide data, see from_wide_data

        This function calculates PME values for each entity in the private equity fund using the provided DataFrame.

        The PME calculation requires information on contributions, distributions, fair market values (FMV), and entities.

        Parameters
        ----------
        pe_fund : pandas.DataFrame
            A DataFrame in long format containing information about the private equity fund.
        index_list : list
            A list of public indices to help PME computations.
        portfolio_list : list[str] or 
            A list of portfolios to compute PME for. 
        contribution_col : str
            The column name containing contribution values.
        distribution_col : str
            The column name containing distribution values.
        fmv_col : str, optional, default: "fmv"
            The column name containing fair market values.
        entity_col : str, optional, default: "entity"
            The column name containing entity information.
        discount_rate : float, optional, default: 0.2
            The discount rate used in the PME calculation.

        Returns
        -------
        PmeTable
            Return an instance of PmeTable.
        """

        if isinstance(pe_fund, dict):
            pe_fund = pd.concat(pe_fund, axis=0)
        elif not isinstance(pe_fund, pd.DataFrame):
            raise TypeError("Input data must be dict of DataFrames or DataFrame") 

        if entity_col not in pe_fund.columns.tolist():
            raise ValueError("You must provide a column from the dataframe from which to subset funds")

        #Creates the receiving array
        res = np.zeros((len(index_list), len(portfolio_list), 2))

        #Check all possible combination of funds and public indexes and recalculate PME for them
        for ind_col, portfolio_iteration in enumerate(portfolio_list): 

            for ind_row, index_iteration in enumerate(index_list):

                full_fund_df = pe_fund.loc[lambda x : (x[entity_col] == portfolio_iteration) & (x[index_iteration] != 0)]
                
                pme = PmeMetrics(full_fund_df[contribution_col], full_fund_df[distribution_col], full_fund_df[fmv_col], full_fund_df[index_iteration], discount_rate=discount_rate)

                res[ind_row, ind_col, :] = [pme.compute_direct_alpha(), pme.compute_ks_pme()]

        #Instantiate a PmeTable
        pme_table = PmeTable(res, index_list, portfolio_list)

        pme_table.table = pme_table.create_plotly_table(res, metric = metric)

        return pme_table
    
    @classmethod
    def from_wide_data(cls, 
                       pe_fund: pd.DataFrame,  
                       index_list: list, 
                       date: str, 
                       contribution_suffix: str, 
                       distribution_suffix: str, 
                       fmv_suffix: str, 
                       portfolio_list : list = None,
                       metric : str = "kspme",
                       discount_rate: float = 0.2) -> "PmeTable":
        """
        Class constructor for PmeTable

        Generate a Public Market Equivalent (PME) table.
        
        Only works with wide data. Requires a specific structure where each column is a combination
        of suffix or prefix and a fund name e.g. fmv_AIFII, contributions_AIFII etc.

        We need this structure to be able to regroup columns by type (Fmv, distribution, etc.) and add a column to identify funds

        Reshapes the data in a long format and then calls from_long_data

        This function calculates PME values for each entity in the private equity fund using the provided DataFrame.

        The PME calculation requires information on contributions, distributions, fair market values (FMV).

        Parameters
        ----------
        pe_fund : pandas.DataFrame
            A DataFrame in wide format containing information about the private equity fund.
        index_list : list
            A list of public indices to help PME computations.
        date: str
            The name of a date column
        contribution_suffix : str
            The identifier of contribution columns, e.g. "contribution_"
        distribution_suffix : str
            The identifier of distribution columns, e.g. "distribution_"
        fmv_suffix : str
            The identifier of fmv columns, e.g. "fmv_"
        portfolio_list : list, optional, by default None
            The list of portfolios on which to compute statistics 
            If portfolio_list ==  None, will return for all funds present in the DataFrame
        metric : str, optional, default: "kspme"
            The metric to produce, 
        discount_rate : float, optional, default: 0.2
            The discount rate used in the PME calculation.

        Returns
        -------
        PmeTable
            Return an instance of PmeTable.

        """
        #Concatenate along the x axis
        if isinstance(pe_fund, dict):
            pe_fund = pd.concat(pe_fund, axis=1)
        elif not isinstance(pe_fund, pd.DataFrame):
            raise TypeError("Input data must be dict of DataFrames or DataFrame") 

        if date not in pe_fund.columns.tolist():
            raise ValueError("You must provide a column from the dataframe to use as date")

        suffixes = (contribution_suffix, distribution_suffix, fmv_suffix)

        #Reshaping from wide to long by type of columns
        full_long_df = []
        for suffix in suffixes:
            
            #Filtering on what columns will be reshaped in long format. For each iteration, columns of the same group (e.g. FMV) are reshaped together
            cols = pe_fund.filter(like=suffix, axis=1).columns.to_list()
            long_df = pd.melt(pe_fund, id_vars=[date], value_vars=cols, var_name='entity', value_name=suffix)

            #Cleaning the newly created entity column to remove the suffix
            long_df["entity"] = long_df["entity"].str.replace(suffix, "")
            full_long_df.append(long_df)
        
        #Recursive merging of the dataframes
        full_long_df = reduce(lambda left, right : pd.merge(left, right, on=[date, "entity"], how='outer'), full_long_df)

        #We recreated a long df from the wide df 
        if portfolio_list is None: 
            portfolio_list = list(full_long_df["entity"].unique())

        #Fusing back the public indexes using many to one relationship to multiply the public indexes 
        full_long_df = (full_long_df.merge(pe_fund.loc[:, index_list + [date]], on = date, how = "left")
                                    .sort_values(["entity", "quarterDate"])
                                    .dropna())
        
        #Then calling the long format method 
        return cls.from_long_data(full_long_df, 
                                  ndex_list, 
                                  portfolio_list, 
                                  contribution_suffix, 
                                  distribution_suffix, 
                                  fmv_suffix, 
                                  entity_col = 'entity', 
                                  discount_rate=0, 
                                  metric = metric)


    def create_plotly_table(self, data: np.ndarray, metric: str = "kspme"):
        """
        Creates a plotly table of PE fund's Public Market Equivalent

        Parameters
        ----------
        data : dict
            A dictionnary of np.arrays with metrics as keys. Each array must have as values the Public Market Equivalent for a combination of PE fund and public index
        index : list
            Axis 0 of the table
        columns : list
            Axis 1 of the table 
        metric : str, optional
            The metric in the table, by default "kspme"
            Can be one of ('direct_alpha', 'kspme')

        Returns
        -------
        go.Figure
            A plotly table of Public Market Equivalent performances 
        """
        #TODO Consider expanding to more metrics 

        #Set for the color_maker. ALSO used as a way to choose the metric since they are stored in axis 2 at values 0 and 1
        threshold = 1
        if metric == "direct_alpha":
            threshold = 0

        #Since the data are stored in a 3D array, retrieves the right metric along the z axis
        data = data[:,:,threshold]

        columns, index = self.index_list, self.portfolio_list
        
        return _create_plotly_table(data, columns, index, threshold = threshold, metric=metric)

