import pandas as pd
import numpy as np
from collections.abc import Callable

class PreprocessData():
    """
    Class centralising preprocessing methods for the VAR dataset 

    Instances of the class contain a list of portfolios on which analysis can be performed. 
    """

    def __init__(self):

        self.funds_size = {
                "AIF I": 200 * 10 ** 6,
                "AIF II": 664 * 10 ** 6,
                "AIF III": 1453 * 10 ** 6,
                "AIF IV": 2650 * 10 ** 6,
                "AIF V": 6146 * 10 ** 6,
                "AAIF IV": (770 * 10 ** 6) * 1.1, #Forex rescale EURUSD
                "AAIF V": (2085 * 10 ** 6) * 1.1, #Forex rescale EURUSD
                "ACEEF": 1002 * 10 ** 6,
                }
                
        #Filled in preprocess
        self.processed_funds = None
        self.processed_benchmarks = None

    def process_funds(self, 
                    metrics : pd.DataFrame, 
                    portfolio_col: str="entity", 
                    funds: str | list[str]=None,
                    inplace: bool=True,) -> "PreprocessData":
        """
        Preprocess metrics DataFrame for VAR analysis. 

        Combines the various files loaded from the lake and cleans them. 

        Subsets the metrics dataframe for only the Ardian's funds. 

        Standardises the date column

        Filters out the non - numerical columns

        Returns a dict of dataframe or a dataframe depending on 
        if there is more than one fund in the dataframe

        Parameters
        ----------
        metrics : pd.DataFrame
            The dataframe to process
        portfolio_col : str
            The column in which the portfolio identifiers are located
        funds: str or list[str]
            The fund(s) to analyse, by default None
            If None, will select all unique values in metrics[portfolio_col].

        Returns
        -------
        PreprocessData
            The instance of PreprocessData with self.processed_funds filled in
        """
        flag_check = True
        unique_funds_list = list(metrics[portfolio_col].unique())
        
        if funds is None:
            funds = unique_funds_list
            flag_check = False #No need to check if we used the available funds

        if isinstance(funds, str):
            funds = [funds]
            
        if flag_check: 
            for fund in funds:
                if fund not in unique_funds_list:
                    raise ValueError(f"{fund} not in existing funds in the dataframe")
                
        #Subset to keep only the fund level rows
        #Creates a time column on which we can groupby
        #Sets index with quarterDate and entity and remove all non-numerical columns
        metrics = (metrics.loc[lambda x: x["entity"].isin(funds),:]
                        .assign(quarterDate = metrics["quarterDate"].dt.to_period("Q").astype(str),)
                        .set_index(["quarterDate", "entity"])
                        .select_dtypes(np.number)
                        .reset_index(level='entity')
                        )        
        cols_to_shift = [col for col in metrics.columns if col[:5] == "total"]
        funds_iterator = (self.col_diff(metrics.loc[metrics[portfolio_col] == fund,:], cols_to_shift) 
                          for fund in funds
                          )
        processed_funds = pd.concat(funds_iterator, axis=0, ignore_index=False)
        
        self.processed_funds = processed_funds
        
        return self
    
    @staticmethod
    def col_diff(df_funds: pd.DataFrame, 
                 cols_to_shift : list[str],) -> pd.DataFrame:
        """
        Creates delta based on measures of total amounts in the dataframe

        Parameters
        ----------
        df_funds : pd.DataFrame
            A dataframe of PE fund performance 

        Returns
        -------
        pd.DataFrame
            A dataframe of PE fund performance with additional delta columns  
        """
        
        shifted_df = (df_funds[cols_to_shift].diff(1)
                                             .rename(columns={col : "".join((col, "Delta")) for col in cols_to_shift})
                                             )
        shifted_df.loc[df_funds.index[0], shifted_df.columns.to_list()] = df_funds.loc[df_funds.index[0], cols_to_shift]
        
        return df_funds.join(shifted_df, how="left")
    
    def process_benchmarks(self, 
                           public_indexes_df : pd.DataFrame,
                           inplace: bool=True,
                           safe_returns_kwargs: dict=None) -> "PreprocessData":
        """
        Merges the PE fund dataframe with the public indexes

        Parameters
        ----------
        public_indexes_df: pd.DataFrame
            DataFrame containing the public indexes

        Returns
        -------
        pd.DataFrame or dict(fund:pd.DataFrame)
            Combination of the PE fund data and the Public indexes data.
            Can be a DataFrame or a dict of DataFrames if several PE funds are selected
            If dict, the format is format {fund_name:joined_data}
            fund_name is a string in ("AIF II", "AIF III", "AIF IV", "AIF V", "AAIF IV", "AAIF V", "ACEEF")
        """
        if safe_returns_kwargs is None:
            safe_returns_kwargs = {"rate":0.02, #Default values 2% annual returns and quarterly periods
                                   "periods":4,
                                   "new_name":"safeReturns",
                                   }
        if not isinstance(safe_returns_kwargs, dict):
            raise TypeError("safe_returns_kwargs must be a dict")
        
        rolled_indexes_df = (public_indexes_df.sort_index()
                                            .rolling(5)    
                                            .mean()
                                            .reset_index()
                                            .assign(quarterDate = lambda x : x["date"].dt.to_period("Q").astype(str).to_numpy())
                                            )
        quarterly_public_returns = (rolled_indexes_df.groupby("quarterDate")
                                                     .aggregate({col: "last" for col in rolled_indexes_df.columns})
                                                     .sort_index()
                                                     .drop(columns=["date", "quarterDate"])
                                                     )
        quarterly_public_returns[safe_returns_kwargs.get("new_name", "safeReturns")] = self.compute_safe_returns(annual_rate=safe_returns_kwargs.get("rate",0.02), 
                                                                                                                 N_steps=len(quarterly_public_returns), 
                                                                                                                 periods=safe_returns_kwargs.get("periods",4)
                                                                                                                 )
        if inplace:
            self.processed_benchmarks = quarterly_public_returns
            return self
        
        return quarterly_public_returns
    
    def compute_new_view(self,
                        scope : list | tuple,
                        new_view : str,
                        metric_df: pd.DataFrame=None,
                        portfolio_col: str="entity",
                        contribution_col: str="investmentsOfTheQuarter", 
                        distribution_col: str="proceedsOfTheQuarter", 
                        time_col: str="quarterDate",
                        fmv_col: str="fmv",
                        is_asset_view: bool=False,
                        assets_kwargs: dict=None,
                        is_deflated: bool=True,
                        inplace: bool=True) -> "PreprocessData":
        
        if metric_df is None:
            metric_df = self.processed_funds.copy()
        
        mask_view = np.logical_not(metric_df[portfolio_col] == new_view).to_numpy()
        mask_scope = metric_df[portfolio_col].isin(scope).to_numpy()
        
        if is_asset_view:
            if assets_kwargs is None: 
                assets_kwargs = {"fundSize_col":"portfolioName"}
            metric_df = (metric_df.loc[mask_view & mask_scope,:]
                                  .assign(fund_size=lambda x : x[assets_kwargs.get("fundSize_col", "portfolioName")].map(self.funds_size))
                                  )
        else:
            metric_df = (metric_df.loc[mask_view & mask_scope,:]
                                  .assign(fund_size=lambda x : x[portfolio_col].map(self.funds_size))
                                  )        
        old_cols = [contribution_col, distribution_col, fmv_col]
        new_cols = ["".join((col, "Deflated")) for col in old_cols]

        if is_deflated:
            metric_df.loc[:,new_cols] = (metric_df.loc[:,old_cols].to_numpy().T / metric_df["fund_size"].to_numpy()).T
        else:
            metric_df.loc[:,new_cols] = metric_df.loc[:,old_cols].copy()
        
        agg_dict = {old_col:(new_col,"sum") 
                    for old_col, new_col in zip(old_cols, new_cols)
                    }
        metrics_pew = (metric_df.groupby(time_col)
                            .agg(**agg_dict)
                            .assign(entity=new_view)
                            )
        metrics_pew["return"] = (self.recalculate_returns(funds_df=metrics_pew, portfolio_col="entity", r_suffix="New", inplace=False)
                                     .loc[:,"returnNew"]
                                     .to_numpy() 
                                     )
        metrics_pew["chronologicalOrder"] = range(metrics_pew.shape[0])
        old_df = self.processed_funds.copy()
        old_df = old_df.loc[~(old_df[portfolio_col] == new_view),:]
        new_df = pd.concat((old_df, metrics_pew), axis=0)

        if inplace: 
            self.processed_funds = new_df.copy()
            return self
        
        return new_df

    def combine_financials(self, 
                           metrics: pd.DataFrame=None, 
                           public_indexes: pd.DataFrame=None,) -> pd.DataFrame:
        
        if metrics is None:
            metrics = self.processed_funds.copy()
        if public_indexes is None:
            public_indexes = self.processed_benchmarks.copy()
        
        return metrics.join(public_indexes, how="left")         
    
    def calculate_returns(self, 
                          public_indexes_df: pd.DataFrame=None,
                          periods: int=1) -> pd.DataFrame:
        
        if public_indexes_df is None: 
            public_indexes_df = self.processed_benchmarks.copy()
        
        return (public_indexes_df.pct_change(periods=periods)
                                 .fillna(0)
                                 )
        
    def rescale_returns(self, 
                        funds_df: pd.DataFrame=None, 
                        return_col: str="return", 
                        transformation_func: Callable=None,
                        r_suffix: str=None,
                        inplace: bool=True) -> "PreprocessData":
        
        if transformation_func is None: 
            def transformation_func(x): 
                x =  x - 0.01 #Removing 1% as management fee
                x[x > 0] = x[x > 0] * 0.85 #Taking 15% of returns as performance fees for positive performances
                return x
        if funds_df is None:
            funds_df = self.processed_funds.copy()
        if r_suffix is None: 
            r_suffix = ""
            
        new_col_name = "".join((return_col, r_suffix))
        funds_df[new_col_name] = transformation_func(funds_df[return_col].to_numpy())
        
        if inplace: 
            self.processed_funds = funds_df.copy()
            return self
        
        return funds_df
    
    def rescale_cashflows(self,
                          funds_df: pd.DataFrame=None, 
                          contributions: tuple[str, float]=("investmentsOfTheQuarter", 0.01),
                          distributions: tuple[str, float]=("proceedsOfTheQuarter", 0.15),
                          r_suffix: str=None,
                          inplace: bool=True) -> "PreprocessData":
    
        if funds_df is None: 
            funds_df = self.processed_funds.copy()
        if r_suffix is None: #Will erase the initial column if r_suffix is None
            r_suffix=""

        contributions_col, contributions_scaler = contributions #unload the tuples
        distributions_col, distributions_scaler = distributions
        distributions_scaler = np.where(funds_df[distributions_col].to_numpy() > 0, distributions_scaler, 0)
        
        new_col_name = "".join((contributions_col, r_suffix))
        funds_df[new_col_name] = funds_df[contributions_col].to_numpy() * (1 + contributions_scaler)
        new_col_name = "".join((distributions_col, r_suffix))
        funds_df[new_col_name] = funds_df[distributions_col].to_numpy() * (1 - distributions_scaler)
        
        if inplace:
            self.processed_funds = funds_df.copy()
            return self    
        
        return funds_df
    
    def recalculate_returns(self, 
                            funds_df: pd.DataFrame=None,
                            portfolio_col: str="entity", 
                            contribution_col: str="investmentsOfTheQuarter", 
                            distribution_col: str="proceedsOfTheQuarter", 
                            fmv_col: str="fmv",
                            r_suffix: str=None,
                            inplace: bool=True) -> "PreprocessData":
        
        if funds_df is None:
            funds_df = self.processed_funds.copy()
        if r_suffix is None: #Will erase the initial column if r_suffix is None 
            r_suffix = ""
        
        new_col_name = "".join(("return", r_suffix))
        funds_df["shiftedFmv"] = funds_df.groupby(portfolio_col)[fmv_col].shift(1)
        funds_df["deltaFmv"] = (funds_df[fmv_col] - funds_df["shiftedFmv"]).fillna(0)
        funds_df["shiftedFmv"] = funds_df["shiftedFmv"].fillna(1)
        funds_df[new_col_name] = ((funds_df["deltaFmv"] + funds_df[distribution_col] - funds_df[contribution_col]) 
                                         / funds_df["shiftedFmv"]
                                         )
        if inplace:
            self.processed_funds = (funds_df.copy()
                                            .drop(columns = ["shiftedFmv", "deltaFmv"])
                                            )
            return self
        
        return funds_df
    
    @staticmethod
    def compute_safe_returns(annual_rate : float, N_steps : int, periods: int=4):  

        period_rate = (1 + annual_rate) ** (1/periods)

        safe_returns = np.full(N_steps, period_rate) #NOTE Check on this one
        safe_returns[0] = 1
        safe_returns = np.cumprod(safe_returns)

        return safe_returns


def fund_scaler(serie: pd.Series, deflator_col: pd.Series) -> pd.DataFrame:
    """
    Rescales cashflows based on last fmv per fund to calculate Portfolio equal weight 

    Parameters
    ----------
    serie : pd.Series
        The contributions to rescale
    deflator_col : str, optional
        The column used to deflate cashflows

    Returns
    -------
    pd.DataFrame
        A dataframe with three added columns of rescaled values
    """

    deflator = np.max(np.array(deflator_col))

    return np.array(serie)/deflator


