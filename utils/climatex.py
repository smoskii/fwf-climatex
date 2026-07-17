#!/Users/cpearson/miniconda3/envs/fwf/bin/python

"""
Transforms ClimateX Data to the WRF domain and matches the variable naming convention used in the FWF class
"""

import context
import json
import salem
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from datetime import datetime
from pathlib import Path

from context import data_dir, root_dir
from utils.diagnostic import solve_RH, solve_W_WD
from utils.formate import formate
from utils.compressor import compressor

def transform_climatex(ds):

    print(ds.Times)

    fwf_ds = xr.Dataset()

    # Rename variables and change units
    fwf_ds["T"] = ds["T2"] - 273.15
    fwf_ds["U10"] = ds["U10"]
    fwf_ds["V10"] = ds["V10"]

    fwf_ds["SNOWH"] = ds["SNOWH"]
    fwf_ds["SNOWC"] = ds["SNOWC"]
    fwf_ds["SNW"] = ds["SNOW"]

    # Calculate accumulated precipitation
    precip = (
    ds.RAINC +
    ds.RAINNC +
    ds.RAINSH
    )

    # WRF precipitation variables (RAINC, RAINNC, RAINSH) are cumulative from model initialization.
    # Remove the initial accumulated value to match the operational FWF input convention.
    fwf_ds["r_o"] = precip - precip.isel(Time=0)

    

    # Calculate wind speed and direction
    fwf_ds["W"] = np.sqrt(
    fwf_ds.U10**2 + fwf_ds.V10**2
    )

    fwf_ds["WD"] = (
        270 - np.rad2deg(
            np.arctan2(fwf_ds.V10, fwf_ds.U10)
        )
    ) % 360

    # Calculate dewpoint from Q2
    q = ds.Q2
    p = ds.PSFC / 100

    e = (q*p)/(0.622 + 0.378*q)

    ln_ratio = np.log(e/6.112)

    fwf_ds["TD"] = (
        237.3 * ln_ratio /
        (17.27 - ln_ratio)
    )

    # Calculate relative humidity from T and TD
    T = fwf_ds.T
    TD = fwf_ds.TD

    RH = (
        6.11 * 10**(7.5*TD/(237.7+TD))
    ) / (
        6.11 * 10**(7.5*T/(237.7+T))
    ) * 100

    fwf_ds["H"] = xr.where(RH > 100, 100, RH)

    # Add WRF coordinates
    fwf_ds = fwf_ds.assign_coords(
    XLAT=(("south_north", "west_east"), ds.XLAT.isel(Time=0).values),
    XLONG=(("south_north", "west_east"), ds.XLONG.isel(Time=0).values)
    )


    time = pd.to_datetime(
    ds.Times.values.astype(str),
    format="%Y-%m-%d_%H:%M:%S"
    )

    fwf_ds = fwf_ds.assign_coords(
        Time=("Time", time)
    )

    return fwf_ds