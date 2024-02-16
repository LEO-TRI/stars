import numpy as np
import pandas as pd 
import functools as ft
import itertools as it
import plotly.graph_objects as go

from .pme_metrics import PmeMetrics
from .financial_graphs import _create_plotly_table

class PmeTable(PmeMetrics):
    """
    Class to calculate and hold a table of PME metrics. 

    Should not be instantiated directly. Instead, use class constructors from_wide_data and 
    from_long_data.  
    """
    def __init__(self, table_data : np.ndarray, index_list : list, portfolio_list : list, table: go.Figure = None):
        
        self.table_data = table_data
        self.index_list = index_list
        self.portfolio_list = portfolio_list

        #Here as a reminder that self.table will be filled later 
        self.table = table 

    @classmethod
    def from_long_data(cls, 
                        pe_fund : pd.DataFrame, 
                        index_list : list, 
                        portfolio_list : list[str], 
                        contribution_col : str, 
                        distribution_col : str, 
                        fmv_col: str="fmv", 
                        daily_data: pd.Series=None, 
                        entity_col: str="entity", 
                        metric: str="kspme",
                        periods: int=4,
                        discount_rate: float=0) -> "PmeTable":
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
        portfolio_list : list[str] 
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
        res = np.empty((len(index_list), len(portfolio_list), 3))
        
        #TODO Find a better way to fill this array with only one loop
        #combination_list = list(it.product(portfolio_list, index_list))

        #Check all possible combination of funds and public indexes and recalculate PME for them
        for ind_col, portfolio_iteration in enumerate(portfolio_list): 

            for ind_row, index_iteration in enumerate(index_list):

                #Subset for the correct Ardian fund and remove all 0s and nan from the benchmark values
                #TODO 
                full_fund_df = pe_fund.loc[lambda x : (x[entity_col] == portfolio_iteration) & ~(x[index_iteration] == 0) & ~(x[index_iteration].isna()),:] 
                
                pme = PmeMetrics(full_fund_df[contribution_col], full_fund_df[distribution_col], full_fund_df[fmv_col], full_fund_df[index_iteration], daily_data=daily_data, discount_rate=discount_rate, periods=periods)

                #Store the 3 metrics across dimension 2 for a given combination of fund and benchmark (respectively dimensions 1 and 0)
                res[ind_row, ind_col, :] = [pme.compute_direct_alpha(), pme.compute_ks_pme(), pme.compute_ln_pme()]

        #Instantiate a PmeTable
        table = PmeTable(res, index_list, portfolio_list).create_plotly_table(metric = metric)

        return PmeTable(res, index_list, portfolio_list, table)
    
    @classmethod
    def from_wide_data(cls, 
                       pe_fund : pd.DataFrame,  
                       index_list : list, 
                       date : str, 
                       contribution_suffix : str, 
                       distribution_suffix : str, 
                       fmv_suffix : str, 
                       portfolio_list: list=None,
                       metric: str="kspme",
                       discount_rate: float=0.2) -> "PmeTable":
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
        pe_fund : pd.DataFrame
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
        full_long_list = []
        for suffix in suffixes:
            
            #Filtering on what columns will be reshaped in long format. For each iteration, columns of the same group (e.g. FMV) are reshaped together
            cols = pe_fund.filter(like=suffix, axis=1).columns.to_list()
            long_df = pd.melt(pe_fund, id_vars=[date], value_vars=cols, var_name='entity', value_name=suffix)

            #Cleaning the newly created entity column to remove the suffix
            long_df["entity"] = long_df["entity"].str.replace(suffix, "")
            full_long_list.append(long_df)
        
        #Recursive merging of the dataframes
        full_long_df = ft.reduce(lambda left, right : pd.merge(left, right, on=[date, "entity"], how='outer'), full_long_list)

        #We recreated a long df from the wide df 
        if portfolio_list is None: 
            portfolio_list = list(full_long_df["entity"].unique())

        #Fusing back the public indexes using many to one relationship to multiply the public indexes 
        full_long_df = (full_long_df.merge(pe_fund.loc[:, index_list + [date]], on = date, how = "left")
                                    .sort_values(["entity", "quarterDate"])
                                    .dropna()
                                    )
        
        #Then calling the long format method 
        return cls.from_long_data(full_long_df, 
                                  index_list, 
                                  portfolio_list, 
                                  contribution_suffix, 
                                  distribution_suffix, 
                                  fmv_suffix, 
                                  entity_col = 'entity', 
                                  discount_rate=0, 
                                  metric = metric)


    def create_plotly_table(self, metric: str = "kspme") -> go.Figure:
        """
        Creates a plotly table of PE fund's Public Market Equivalent

        Parameters
        ----------
        index : list
            Axis 0 of the table
        columns : list
            Axis 1 of the table 
        metric : str, optional
            The metric in the table, by default "kspme"
            Can be one of ('direct_alpha', 'kspme', "ln_pm")

        Returns
        -------
        go.Figure
            A plotly table of Public Market Equivalent performances 
        """
        
        data_z_dim = dict(direct_alpha=0, kspme=1, lnpme=2)
        metric_key = data_z_dim.get(metric)

        #Since the data are stored in a 3D array, retrieves the right metric along the z axis
        data = self.table_data[:,:,metric_key]
        
        #Set for the color_maker. For LNPME and Direct Alpha, funds' outperformance are measured 
        #when above 0 and in %. For the KSPME, fund outperformance is above 1.
        if metric == "kspme":
            threshold = 1
        else : 
            threshold = 0
            data = data * 100

        columns, index = self.index_list, self.portfolio_list
        
        return _create_plotly_table(data, columns, index, threshold = threshold, metric=metric)