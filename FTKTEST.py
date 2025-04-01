from datetime import datetime
import importlib
import numpy as np
import pandas as pd
from scipy.stats import norm

import bql
from models.tia.bbg import LocalTerminal
from datetime import datetime
from Calendars import calendars
import XBond as xb
importlib.reload(xb)
import XCTK
import XBTK
from dataHandling.getDLVBaskets import GetDLVBasket
import re

bq = bql.Service()

import os
import sys

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(os.path.abspath('')))
parent_dir = os.path.dirname(os.path.realpath(parent_dir))
sys.path.append('dataHandling')

FuturesMetaData = {
    'Index': ['ccy', 'rfrCurve', 'tsyCurve', 'pcs', 'mmktDC', 'otr'],
    'TU': ['USD', 'S0490', 'YCGT0025', 'BVN3', 360, 'USGG2YR Index'],
    'FV': ['USD', 'S0490', 'YCGT0025', 'BVN3', 360, 'USGG5YR Index'],
    'TY': ['USD', 'S0490', 'YCGT0025', 'BVN3', 360, 'USGG7YR Index'],
    'UXY': ['USD', 'S0490', 'YCGT0025', 'BVN3', 360, 'USGG15YR Index'],
    'US': ['USD', 'S0490', 'YCGT0025', 'BVN3', 360, 'USGG15YR Index'],
    'WN': ['USD', 'S0490', 'YCGT0025', 'BVN3', 360, 'USGG25YR Index'],
}

FuturesMetaData = pd.DataFrame(FuturesMetaData).set_index('Index').T

# Futures basket data with details of each bond and which futures contract they map to.
dlv_fut_data = pd.read_hdf('../dataHandling/futures.h5', 'dlv_fut_data')

# Mapping table of options contracts to futures contracts.
opt_fut_data = pd.read_hdf('../dataHandling/futures.h5','opt_fut_data')

# Futures data with details of the futures expiry, first notice, first delivery and last delivery.
fut_data = pd.read_hdf('../dataHandling/futures.h5','fut_data')

class BondFuture:
    
    def __init__(self, contract : str):
        """Constructs a treasury bond futures contract.  Not applicable to cash settled futures like ASX or KTB Futures.

        Parameters
        ----------
        contract : str
            The Bloomberg ticker of the contract without the Yellow Key.
        """
        # Set futures attributes
        self.contract = contract
        self.ticker = contract + ' COMB Comdty'

        # Look up contract details from metadata table.  Get corresponding, currency, rfr curve, on-the-run curve and money market day count convention.
        # This is for the GENERIC contract, not data relative to a SPECIFIC contract.
        metadata = FuturesMetaData[FuturesMetaData.index.str.startswith(contract[:2])].reset_index()
        self.ccy = metadata.loc[0, 'ccy']
        self.rfrCurve = metadata.loc[0, 'rfrCurve']
        self.tsyCurve = metadata.loc[0, 'tsyCurve']
        self.pcs = metadata.loc[0, 'pcs']
        self.mmkt_day_count = metadata.loc[0, 'mmktDC']

        # Get data related to the SPECIFIC contract, e.g. U4, Z4 or H5.
        contract_data = fut_data[fut_data['Futures Contract'] == contract].reset_index(drop = True)
        self.first_notice = np.datetime64(contract_data.loc[0, 'FUT_NOTICE_FIRST'], 'D')
        self.first_delivery = np.datetime64(contract_data.loc[0, 'first_delivery'], 'D')
        self.futures_expiration = np.datetime64(contract_data.loc[0, 'last_trade'], 'D')
        self.last_delivery = np.datetime64(contract_data.loc[0, 'last_delivery'], 'D')
        self.px_last = None
        self.DV01 = None
        self.lognorm_vol = None
        self.nvol = None
        
        # Set the DataFrame to store the bond data by default to the current bond basket stored in the database.
        self.bond_data = dlv_fut_data[dlv_fut_data['Futures Contract'] == self.contract].reset_index(drop = True)
        self._bond_data_cols = list(dlv_fut_data.columns)
        
        # Default the trade date and exercise date to NA.
        self.tradeDate = np.datetime64('NaT')
        self.exerciseDate = np.datetime64('NaT')

    def _update_DLV_basket(self, tradeDate, actualRepo):
        # Store a temporary copy of the bond data
        bond_data = self.bond_data.copy()[self._bond_data_cols]

        # If trade date is today, or different to last provided trade date, refresh the bond curve and deliverable basket
        if tradeDate != str(self.tradeDate)[:10] or tradeDate == np.datetime64('today'):
            # self.bondCurve = XBTK.BondCurve(curve = self.tsyCurve, curveDate = self.tradeDate)
            self.tradeDate = np.datetime64(tradeDate)

            # Get the trade date of the stored basket data.
            basket_date = bond_data.loc[0, 'trade_date']

            # Strip the on-the-run curve
            self.bond_curve = XBTK.BondCurve(curve = self.tsyCurve, curveDate = str(tradeDate)[:10])

            # If the trade data of the stored basket does not match the trade date, pull the basket on the trade date
            if np.datetime64(basket_date) != np.datetime64(tradeDate):
                futures_tickers = [self.contract + ' COMB Comdty']
                #! Need to adjust to handle the trade date.  Just pulling the latest data at this point.
                bond_data = GetDLVBasket(tickers = futures_tickers, file_path = 'dataHandling/futures.h5', trade_date = tradeDate, return_data = True)[2]
                # Filter the return data for the contract, #! Not really required.
        
        bond_data = bond_data[bond_data['Futures Contract'] == self.contract].reset_index(drop = True)

        # Request price and yield data from Bloomberg for the trade date.
        bq_univ = bq.univ.list(bond_data['ID'].drop_duplicates().to_list() + [self.ticker])

        # Get the prices source for the bonds.  If the trade date is today use BGN, otherwise use the futures snap.
        pcs_src = 'BGN' if np.datetime64(tradeDate) == np.datetime64('today') else self.pcs

        # Build the request for the price and yield data on the bonds and futures.
        flds = {'price' : bq.func.dropna(bq.data.px_last(dates = str(tradeDate)[:10], PRICING_SOURCE = pcs_src)),
                'yield' : bq.func.dropna(bq.data.yield_(side = 'Mid', dates = str(tradeDate)[:10], YIELD_TYPE='YTW', PRICING_SOURCE = pcs_src))}
        
        req = bql.Request(bq_univ, flds)

        resp = bq.execute(req)

        # Dat_tsr is used to parse static history data while ignoring static data which is not present in the static history data fields
        dat_tsr = pd.DataFrame({r.name:r.df().reset_index().set_index(['ID', 'DATE'])[r.name] for r in resp}).reset_index().dropna(subset = ['DATE']).reset_index(drop = True)
        
        # Map the futures price to the bond basket data
        bond_data['futures_price'] = dat_tsr.set_index('ID').loc[self.contract + ' COMB Comdty', 'price']
        self.px_last = bond_data['futures_price'].iloc[0]

        # Merge the price data with the basket data.
        bond_data = pd.merge(bond_data, dat_tsr, on = ['ID'])

        bond_data['delivery_first'] = fut_data[fut_data['Futures Contract'] == self.contract]['first_delivery'].iloc[0]
        bond_data['delivery_last'] = fut_data[fut_data['Futures Contract'] == self.contract]['last_delivery'].iloc[0]

        bond_data['actual_repo'] = actualRepo

        bond_data['delivery'] = np.where(bond_data['coupon'].values /365 < bond_data['actual_repo'].values / self.mmkt_day_count, bond_data['delivery_first'].values, bond_data['delivery_last'].values)

        self._calc_contract_risk(tradeDate)

        return bond_data

    def _calc_contract_risk(self, tradeDate):
        tradeDate = re.sub('-', '', tradeDate)

        resp = LocalTerminal.get_historical([self.ticker], ['RK074', 'CONVENTIONAL_CTD_FORWARD_FRSK', 'PX_LAST'], start = tradeDate, end = tradeDate, ignore_field_error = 1)
        fut_data = resp.as_frame()[self.ticker]

        log_vol = fut_data['RK074'].iloc[0]
        DV01 = fut_data['CONVENTIONAL_CTD_FORWARD_FRSK'].iloc[0]
        px_last = fut_data['PX_LAST'].iloc[0]
        nvol = np.sqrt((log_vol ** 2) * (px_last ** 2) / (DV01 ** 2))

        self.lognorm_vol = log_vol
        self.DV01 = DV01
        self.px_last = px_last
        self.nvol = nvol

    #! Clean up to match OADLV
    def DLV(self, tradeDate : str, actualRepo : float, delivery : str = 'last', sort_by : str = 'implied_repo', order : str = 'descending', px_override_map : dict = None, return_basket : bool = True) -> pd.DataFrame:    
        """Calculate the basic cheapest to deliver analytics.

        Parameters
        ----------
        tradeDate : str
            Trade date to run the cheapest-to-deliver analytics
        actualRepo : float
            The repo rate corresponding to the trade horizon
        delivery : str, optional
            The delivery timing to use in calcuations, by default 'last'
        sort_by : str, optional
            The criterion to rank the bonds by, by default 'implied_repo'
        order : str, optional
            Whether to rank the criterion in ascending or descending order, by default 'descending'
        px_override_map : dict, optional
            Dictionary to override the prices for debugging purposes, by default None
        return_basket : bool, optional
            Whether or not to return the DLV basket data, by default True

        Returns
        -------
        pandas.DataFrame
            A table containing the cheapest-to-delivery basket analytics

        Raises
        ------
        ValueError
            Value error raised if invalid delivery time is passed to function.  It must be one of 'last', 'first' or 'optimal'
        """
        # Update the bond data to the trade date if required.
        bond_data = self._update_DLV_basket(tradeDate=tradeDate, actualRepo = actualRepo)

        #! For testing, can remove this after.  Overrides the price and yield data with a hard code for calibration purpoess.
        if px_override_map is not None:
            bond_data['yield0'] = 4.00
            bond_data['price'] = bond_data['isin'].str.slice(0, 11).map(px_override_map)
            bond_data['yield'] = xb.bond_yield_vec(settle = bond_data['settle_date'].to_numpy().astype('datetime64[D]'),
                                                maturity = bond_data['maturity'].to_numpy().astype('datetime64[D]'),
                                                coupon = bond_data['coupon'].values,
                                                price = bond_data['price'].values,
                                                cpn_freq = bond_data['freq'].values,
                                                yield0 = bond_data['yield0'].values,
                                                clean = True,
                                                ex_int = 0)
            del bond_data['yield0']

        # Calculate delivery price and gross basis from the futures price/conversion factor.
        bond_data['delivery_price'] = bond_data['futures_price'] * bond_data['Conversion Factor']
        bond_data['gross_basis'] = bond_data['price'] - bond_data['delivery_price']

        # Set the delivery time to look up the relevant delivery date column depending on whether we want to use first, last of optimal delivery.
        if delivery.lower() == 'optimal':
            delivery_time = 'delivery'
        elif delivery.lower() == 'first':
            delivery_time = 'delivery_first'
        elif delivery.lower() == 'last':
            delivery_time = 'delivery_last'
        else:
            raise ValueError("Delivery must be one of 'optimal', 'first' or 'last'.")

        # Calculate the optimal delivery date for each bond.  To be used if the delivery method is set to be 'optimal'.
        bond_data['delivery'] = np.where(bond_data['coupon'].values /365 < bond_data['actual_repo'].values / self.mmkt_day_count, bond_data['delivery_first'].values, bond_data['delivery_last'].values)

        # Calcuate the implied repo.  Pass the bond data and delivery time to the _irr method.  This needs to be extended to handle cases with multiple delivery dates.
        bond_data['implied_repo'] = self._irr(bond_data, delivery_time)

        # Calculate the forward price of the bonds
        bond_data['forward_price'] = xb.forward_price(settle = bond_data['settle_date'].values.astype('datetime64[D]'), 
                                                      settle_px = bond_data['price'].values, 
                                                      forward_date = bond_data[delivery_time].values.astype('datetime64[D]'), 
                                                      coupon = bond_data['coupon'].values, 
                                                      maturity = bond_data['maturity'].values.astype('datetime64[D]'), 
                                                      actual_repo = actualRepo,
                                                      rfr_curve = None,
                                                      mmkt_day_count = self.mmkt_day_count)
        
        # Difference the spot and forward prices to get the carry
        bond_data['net_carry'] = bond_data['price'] - bond_data['forward_price']

        #! Should make this use the actual forward price
        bond_data['BNOC'] = bond_data['gross_basis'] - bond_data['net_carry']

        # Sort the data to get the CTD
        bond_data = bond_data.sort_values(by = sort_by, ascending = (order.lower() == 'ascending'))

        # Cache the bond data
        self.bond_data = bond_data

        # If the return flag is set to true, return the bond data
        if return_basket:
            return bond_data[['ticker', 'coupon', 'maturity', 'price', 'yield', 'gross_basis', 'implied_repo', 'actual_repo', 'BNOC']]


    #! Need to extend to case with two or more coupons
    def _irr(self, bond_data, delivery_time):

        # Set the relevant data fields to arrays
        settle = bond_data['settle_date'].values.astype('datetime64[D]')
        delivery = bond_data[delivery_time].values.astype('datetime64[D]')
        coupon = bond_data['coupon'].values
        freq = bond_data['freq'].values
        maturity = bond_data['maturity'].values.astype('datetime64[D]')
        price = bond_data['price'].values
        delivery_price = bond_data['delivery_price'].values
        ncd = bond_data['NCD'].values.astype('datetime64[D]')

        # C to be the value of the first coupon paid in the intervening period between the futures settlement date and the delivery date
        c = np.where(delivery < ncd, 0, coupon)

        # Dirty price of the bonds
        dirty_price = price + xb.acc_int(settle = settle, maturity = maturity, coupon = coupon, cpn_freq = freq)
        
        # Dirty invoice prices
        dirty_delivery_price = delivery_price + xb.acc_int(settle = delivery, maturity = maturity, coupon = coupon, cpn_freq = freq)

        # Days between futures settlement and delivery
        n = (delivery - settle).astype(int)

        # Days between the next coupon date and delivery
        n2 = (delivery - ncd).astype(int)

        # Return the implied repo rate, MWRR on the principal and coupon accrued at repo rate on ACT/360 or ACT/365 day count
        return ((dirty_delivery_price + c / freq - dirty_price) * self.mmkt_day_count) / (dirty_price * n - c / freq * n2) * 100


    def OADLV(self, tradeDate : str, actualRepo : float, exerciseDate: str = None, delivery : str = 'last', sort_by : str = 'implied_repo', repo_model : str = 'const_repo', repo_beta : float = 1.00,
              order : str = 'descending', px_override_map : dict = None, return_basket : bool = True, sim_range = 1000):
        """Calculates the option adjusted basis on the deliverable bond basket.

        Parameters
        ----------
        tradeDate : str
            Trade date to run the cheapest-to-deliver analytics
        actualRepo : float
            The repo rate corresponding to the trade horizon
        delivery : str, optional
            The delivery timing to use in calcuations, by default 'last'
        sort_by : str, optional
            The criterion to rank the bonds by, by default 'implied_repo'
        order : str, optional
            Whether to rank the criterion in ascending or descending order, by default 'descending'
        px_override_map : dict, optional
            Dictionary to override the prices for debugging purposes, by default None
        return_basket : bool, optional
            Whether or not to return the DLV basket data, by default True
        sim_range : int, optional
            The range over which to simulate the bond futures price.  Measured in basis points from the current market level, by default 800
        """
        # Set the exercise date
        if exerciseDate is None:
            exerciseDate = np.datetime64(str(tradeDate)[:10])
        else:
            exerciseDate = np.datetime64(str(exerciseDate)[:10])

        # Check if simulation needs to be updated.  If the trade date or exercise date has been updated it will need to be rerun.  Also, if the trade date is today it will need to be 
        # rerun as well to factor in the change in bond and future prices during the day.
        if tradeDate != str(self.tradeDate)[:10] or np.datetime64(tradeDate) == np.datetime64('today') or exerciseDate != self.exerciseDate:
            rerun_sim = True
        else:
            rerun_sim = False
        
        # Run the basic DLV analysis to update all the bond data
        bond_data = self._update_DLV_basket(tradeDate=tradeDate, actualRepo = actualRepo)

        # Cache the exercise date:
        if exerciseDate != self.exerciseDate:
            self.exerciseDate = exerciseDate
        

        #! For testing, can remove this after.  Overrides the price and yield data with a hard code for calibration purpoess.
        if px_override_map is not None:
            bond_data['yield0'] = 4.00
            bond_data['price'] = bond_data['isin'].str.slice(0, 11).map(px_override_map)
            bond_data['yield'] = xb.bond_yield_vec(settle = bond_data['settle_date'].to_numpy().astype('datetime64[D]'),
                                                   maturity = bond_data['maturity'].to_numpy().astype('datetime64[D]'),
                                                   coupon = bond_data['coupon'].values,
                                                   price = bond_data['price'].values,
                                                   cpn_freq = bond_data['freq'].values,
                                                   yield0 = bond_data['yield0'].values,
                                                   clean = True,
                                                   ex_int = 0)
            del bond_data['yield0']

        # Save the latest cut of the bond data
        self.bond_data = bond_data
        
        # Rerun the simulation if either the trade date or exercise date have been updated, or if the trade date is today
        if rerun_sim:
            # Run the bond simulation.  Cache the data in self.bond_sim
            self.DLVSimParallel(tradeDate, exerciseDate, delivery, actualRepo, repo_model, repo_beta, sim_range)

    def DLVSimParallel(self, tradeDate, exerciseDate, delivery, actualRepo, repo_model, repo_beta, sim_range):
        
        bond_data = self.bond_data

        # Set the delivery time to look up the relevant delivery date column depending on whether we want to use first, last of optimal delivery.
        if delivery.lower() == 'optimal':
            delivery_time = 'delivery'
        elif delivery.lower() == 'first':
            delivery_time = 'delivery_first'
        elif delivery.lower() == 'last':
            delivery_time = 'delivery_last'
        else:
            raise ValueError("Delivery must be one of 'optimal', 'first' or 'last'.")
        
        #! Convert these lines into a function
        bond_data['days_to_deliv'] = (bond_data[delivery_time] - bond_data['settle_date']).dt.days
        bond_data['raw_bond_carry'] = bond_data['coupon']  * bond_data['days_to_deliv'] / 365
        bond_data['dirty_price'] = bond_data['price'].values + xb.acc_int(settle = bond_data['settle_date'].values.astype('datetime64[D]'), maturity = bond_data['maturity'].values.astype('datetime64[D]'), 
                                                                          coupon = bond_data['coupon'].values, cpn_freq = bond_data['freq'].values)
        bond_data['raw_repo_carry'] = (bond_data['dirty_price']) * bond_data['actual_repo'] / 100 * bond_data['days_to_deliv'] / self.mmkt_day_count
        bond_data['raw_carry'] = bond_data['raw_bond_carry'] - bond_data['raw_repo_carry']

        # Set the number of shocks, such that the scenarios are a symmetric set of points around the current market level.
        n_shocks = int(sim_range + 1)

        # Generate the shocks as a vector.
        shocks = np.linspace(-sim_range / 2, sim_range / 2, n_shocks)

        # Set the number of shocks, such that the scenarios are a symmetric set of points around the current market level.
        n_shocks = int(sim_range + 1)

        # Generate the shocks as a vector.
        shocks = np.linspace(-sim_range / 2, sim_range / 2, n_shocks)

        bond_data = bond_data.sort_values(by = ['maturity'])

        # Restrip the bond curve across the curve shock scenarios
        z_rates = self.bond_curve.stripCurve(shock = shocks)

        # Compute the coupon dates as a square array and store the output in the DataFrame for better performance.  Also reduces complexity.
        cpn_dates_arr = xb.get_bond_cpn_dates(settle = bond_data['settle_date'].values.astype('datetime64[D]'), 
                                              maturity = bond_data['maturity'].values.astype('datetime64[D]'), 
                                              cpn_freq = bond_data['freq'].values)

        # Convert to times under an ACT/365 day count.  To be used for discounting callcs.  Cases where the time is 0 need to be manually converted back to NaNs.
        cpn_T_arr = (cpn_dates_arr - bond_data['settle_date'].values.astype('datetime64[D]').reshape(-1, 1)).astype(float) / 365
        cpn_T_arr[cpn_T_arr <= 0] = np.nan
        bond_data['cpn_T'] = list(cpn_T_arr)

        bond_data['f'] = np.where(np.isin(bond_data['ticker'], ('B', 'ACTB')), (bond_data['maturity'] - bond_data['settle_date']).dt.days, (bond_data['NCD'] - bond_data['settle_date']).dt.days)
        bond_data['d'] = (bond_data['NCD'] - bond_data['PCD']).dt.days
        bond_data['n'] = xb.coup_n(date_1 = bond_data['settle_date'].values.astype('datetime64[D]'),
                                date_2 = bond_data['maturity'].values.astype('datetime64[D]'),
                                freq = bond_data['freq'].values)
        bond_data['mat_T'] = (bond_data['maturity'] - bond_data['settle_date']).dt.days / 365

        bond_data['z'] = self.bond_curve.z_spread_govy(px = bond_data['price'].values,
                                                coupon = bond_data['coupon'].values,
                                                f = bond_data['f'].values,
                                                d = bond_data['d'].values,
                                                mat_T = bond_data['mat_T'].values,
                                                cpn_T = np.array(list(bond_data['cpn_T'].values)))

        # The bond shock matrix with each bond revalued across all par curve shifts.  The rows are the par curve shocks,
        # The columns are the bonds.
        sim_matrix = self.bond_curve.arb_free_bond_price(coupon = bond_data['coupon'].values,
                                                         z = z_rates,
                                                         f = bond_data['f'].values,
                                                         d = bond_data['d'].values,
                                                         mat_T = bond_data['mat_T'].values,
                                                         cpn_T = np.array(list(bond_data['cpn_T'].values)),
                                                         z_spd_shock = bond_data['z'].values)

        shocks = np.repeat(shocks, len(bond_data))
        bond_sim = pd.concat([bond_data] * n_shocks, ignore_index = True)
        bond_sim['shock'] = shocks

        bond_sim['price'] = sim_matrix.ravel()

        if repo_model.lower() == 'const_carry':
            bond_sim['forward_price'] = bond_sim['price'] - bond_sim['raw_carry']
        elif repo_model.lower() == 'const_repo':
            bond_sim['forward_price'] = xb.forward_price(settle = bond_sim['settle_date'].values.astype('datetime64[D]'), 
                                                                settle_px = bond_sim['price'].values, 
                                                                forward_date = bond_sim[delivery_time].values.astype('datetime64[D]'), 
                                                                coupon = bond_sim['coupon'].values, 
                                                                maturity = bond_sim['maturity'].values.astype('datetime64[D]'), 
                                                                actual_repo = actualRepo,
                                                                mmkt_day_count = self.mmkt_day_count)
        elif repo_model.lower() == 'shift_repo':
            bond_sim['actual_repo'] = actualRepo + bond_sim['shock'].values * repo_beta / 100
            bond_sim['forward_price'] = xb.forward_price(settle = bond_sim['settle_date'].values.astype('datetime64[D]'), 
                                                                settle_px = bond_sim['price'].values, 
                                                                forward_date = bond_sim[delivery_time].values.astype('datetime64[D]'), 
                                                                coupon = bond_sim['coupon'].values, 
                                                                maturity = bond_sim['maturity'].values.astype('datetime64[D]'), 
                                                                actual_repo = bond_sim['actual_repo'].values,
                                                                mmkt_day_count = self.mmkt_day_count)
        
        
        
        # Get the option-free futures price
        bond_sim['opt_free_futures_px'] = bond_sim['forward_price'] / bond_sim['Conversion Factor']

        # Cache the bond_sum data
        self.bond_sim = bond_sim

        # Get the CTD for each simulation
        ctd_data = bond_sim.loc[bond_sim.groupby('shock')['opt_free_futures_px'].idxmin()].reset_index(drop = True)

        # Get the effective duration on the futures notional
        self.eff_dur_notl = (ctd_data[ctd_data['shock'] == -1]['opt_free_futures_px'].values[0] - ctd_data[ctd_data['shock'] == 1]['opt_free_futures_px'].values[0]) / 2 * 100

        # Get the effective duration on the underlying CTD
        self.eff_dur_ctd = self.eff_dur_notl * ctd_data[ctd_data['shock'] == 0]['Conversion Factor'].values[0]

        self.ctd_data = ctd_data

        #! Price the switch option
        self.switch_option(ctd_data, tradeDate, exerciseDate, delivery_time)

        # Reload the bond_sim data with the theo
        bond_sim = self.bond_sim

        # Calculate delivery price and gross basis from the futures price/conversion factor.
        bond_sim['delivery_price'] = bond_sim['theo_futures_px'] * bond_sim['Conversion Factor']
        bond_sim['gross_basis'] = bond_sim['price'] - bond_sim['delivery_price']

        # Calculate the optimal delivery date for each bond.  To be used if the delivery method is set to be 'optimal'.
        bond_sim['delivery'] = np.where(bond_sim['coupon'].values /365 < bond_sim['actual_repo'].values / self.mmkt_day_count, bond_sim['delivery_first'].values, bond_sim['delivery_last'].values)

        # Calcuate the implied repo.  Pass the bond data and delivery time to the _irr method.  This needs to be extended to handle cases with multiple delivery dates.
        bond_sim['implied_repo'] = self._irr(bond_sim, delivery_time)

        # Calc the final carry
        bond_sim['net_carry'] = bond_sim['price'] - bond_sim['forward_price']

        #! Should make this use the actual forward price
        bond_sim['BNOC'] = bond_sim['gross_basis'] - bond_sim['net_carry']

        # Cache the bond_sim data
        self.bond_sim = bond_sim
        
    def switch_option(self, ctd_data, tradeDate, exerciseDate, delivery_time, ctd_distr_rng = 400):
        # Convert the trade date to datetime64
        tradeDate = np.datetime64(tradeDate)

        # Set the number of shocks, such that the scenarios are a symmetric set of points around the current market level.
        k_shocks = int(ctd_distr_rng + 1)

        # Set the distribution range to price the switch option over
        distr_shocks = np.linspace(-ctd_distr_rng / 2, ctd_distr_rng / 2, k_shocks)
        nvol = self.nvol
        expT = (self.last_delivery - exerciseDate).astype(int) / 365
        expT = np.clip(expT, a_min = 0.00000001, a_max = None)
        z_scores = distr_shocks / (nvol * np.sqrt(expT))

        # Get the CDF under the specified yield curve distribution
        cum_prob = norm.cdf(z_scores)

        # Get the approx. PDF under the specified yield curve distribution
        diff_prob = np.diff(cum_prob, prepend=0)

        # Set diff prob to 0 if exercise date equal to last delivery
        if tradeDate == self.last_delivery:
            diff_prob = np.zeros(diff_prob.shape)

        # Get the bond simulation data
        bond_sim = self.bond_sim

        # Get the subset of deliverable bonds, that are cheapest-to-deivery in any of the yield curve scenarios.
        ctd_set = tuple(ctd_data['isin'].drop_duplicates().to_list())

        # Get the option-free futures price for every single scenario, for every single bond in the CTD subset.
        opt_free_ctd_fut_px = bond_sim[bond_sim['isin'].isin(ctd_set)].pivot(index = 'shock', columns = 'isin', values = 'opt_free_futures_px').reset_index()

        # Create column names for the theoretical price of the futures under all scenarios, assuming each bond is the only deliverable bond.
        ctd_theo_cols = [x + '_theo' for x in ctd_set]

        # Get the theoretical price of the future, assuming each CTD bond is the only bond in the deliverable basket.  Used to value the switch option.
        ctd_data[ctd_theo_cols] = opt_free_ctd_fut_px[list(ctd_set)]

        # Create the columns for the distribution of the theoretical futures prices
        ctd_theo_sim_cols = [x + '_theo_sim' for x in ctd_set]

        # Calculate the expected futures prices using the theoretical futures prices which each assume only 1 bond in the deliverable basket
        ctd_data[ctd_theo_sim_cols] = ctd_data[ctd_theo_cols].rolling(window=k_shocks, center = True).apply(lambda x :  np.dot(x, diff_prob), raw=True)

        # Calculate the expected futures price using the option-free futures prices assuming a full deliverable basket
        ctd_data['sim_theo'] = ctd_data['opt_free_futures_px'].rolling(window=k_shocks, center = True).apply(lambda x : np.dot(x, diff_prob), raw=True)

        # Pick out the theoretical futures prices for each scenario, assuming only the CTD is in the basket at any level of yields
        ctd_data['sim_ctd_price'] = ctd_data.apply(lambda row : row[row['isin'] + '_theo_sim'], axis = 1)

        # The switch option value is the difference between these
        ctd_data['switch_option'] = ctd_data['sim_ctd_price'] - ctd_data['sim_theo']

        # Calc the rich/cheap
        oaBasis = ctd_data.set_index(['shock']).loc[0, 'futures_price'] - (ctd_data.set_index(['shock']).loc[0, 'opt_free_futures_px'] - ctd_data.set_index(['shock']).loc[0, 'switch_option'])

        # Scaled the rich cheap down to capture decay between the trade date and exercise date, assume the basis decays linearly...
        self.oaBasis = oaBasis * (1 - (exerciseDate - tradeDate).astype(int) / (self.last_delivery - tradeDate).astype(int))

        # Add back the switch option to get the futures theo
        ctd_data['theo_futures_px'] = ctd_data['opt_free_futures_px'] - ctd_data['switch_option'] + self.oaBasis

        # Also put into the bond_sim DataFrame
        bond_sim['theo_futures_px'] = pd.merge(ctd_data, bond_sim, on = ['shock'], how = 'right')['theo_futures_px']

        ctd_data['delivery_price'] = ctd_data['theo_futures_px'] * ctd_data['Conversion Factor']

        # Calcuate the implied repo.  Pass the bond data and delivery time to the _irr method.  This needs to be extended to handle cases with multiple delivery dates.
        ctd_data['implied_repo'] = self._irr(ctd_data, delivery_time)

        # Store the CTD dat in the CTD attribute
        self.ctd_data = ctd_data


    def CMS(self, tradeDate : str, actualRepo : float, exerciseDate: str = None, delivery : str = 'last', sort_by : str = 'implied_repo', repo_model : str = 'const_repo', repo_beta : float = 1.00,
              order : str = 'descending', px_override_map : dict = None, return_basket : bool = True, sim_range = 1000):
        """Calculates the option adjusted basis on the deliverable bond basket.

        Parameters
        ----------
        tradeDate : str
            Trade date to run the cheapest-to-deliver analytics
        actualRepo : float
            The repo rate corresponding to the trade horizon
        delivery : str, optional
            The delivery timing to use in calcuations, by default 'last'
        sort_by : str, optional
            The criterion to rank the bonds by, by default 'implied_repo'
        order : str, optional
            Whether to rank the criterion in ascending or descending order, by default 'descending'
        px_override_map : dict, optional
            Dictionary to override the prices for debugging purposes, by default None
        return_basket : bool, optional
            Whether or not to return the DLV basket data, by default True
        sim_range : int, optional
            The range over which to simulate the bond futures price.  Measured in basis points from the current market level, by default 1000
        """
        self.OADLV(tradeDate, actualRepo, exerciseDate, delivery, sort_by, repo_model, repo_beta, order, px_override_map, return_basket)
        
        # Reload the bond_sim data with the theo
        bond_sim = self.bond_sim

        # Convert the maturity into a string
        bond_sim['mat_str'] = bond_sim['maturity'].dt.strftime('%d/%m/%Y')
        
        # Generate the ticker coupon maturity
        bond_sim['ticker_cpn_maturity'] = bond_sim.apply(lambda x : x['ticker'] + ' ' + str(x['coupon']) + ' ' + str(x['mat_str']), axis = 1)

        sim_pivot = bond_sim[bond_sim['shock'].isin((-100, -50, 0, 50, 100))].pivot_table(values = 'BNOC', index = 'isin', columns = 'shock').reset_index()
        sim_pivot.index.name = ''



        sim_pivot = sim_pivot.sort_values(by = [0])

        return sim_pivot

    def DLVSimFull(self):
        pass

class ACGBFuture(BondFuture):
    def __init__(self, contract):
        super().__init__self(contract)
