#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import datetime as dt
import xarray as xr
import cdsapi


class SEAS5():
    def __init__(self, cdsapi_key, varmap={}, varmap_da={}):
        """Convenience class for downloading and processing ECMWF SEAS5
        seasonal forecast data.

        Parameters
        ----------
        cdsapi_key : str
            Copernicus CDSAPI key.
        varmap : dict
            Mapping from short to long SEAS5 variable names.
            Default variables include pre, tmax, tmin and sst.
        varmap_da : dict
            Mapping from short to SEAS5 variable names in the downloaded files.
            Default variables include pre, tmax, tmin and sst.
        """

        # List available precomputed climatologies for calculating anomalies
        self.climpath = os.path.join(os.path.dirname(__file__), 'clims')
        self.clims_available = [f.split('.')[0] for f in os.listdir(self.climpath)
                                if not f.startswith('.')]
        self.clim = None

        # Need a Copernicus Data Store Beta (CDS-Beta) API key
        self.c = cdsapi.Client(key=cdsapi_key,
                               url='https://cds-beta.climate.copernicus.eu/api')

        # Mapping from short to long SEAS5 variable names
        self.varmap = {'pre': 'total_precipitation',
                       'tmax': 'maximum_2m_temperature_in_the_last_24_hours',
                       'tmin': 'minimum_2m_temperature_in_the_last_24_hours',
                       'sst': 'sea_surface_temperature',
                       **varmap}

        # Mapping from short to SEAS5 variable names in the DataArray
        self.varmap_da = {'pre': 'tprate',
                          'tmax': 'mx2t24',
                          'tmin': 'mn2t24',
                          'sst': 'sst',
                          **varmap_da}

    def _year_range2years(self, year_range=(None,None), hindcast=False, forecast=True):
        """Convenience function to convert a year_range tuple to a working range
        accounting for hindcast and forecast availability.
        """

        # Define year ranges
        this_year = dt.date.today().year
        years_hindcast = set(range(1981, 2017))
        years_forecast = set(range(2017, this_year+1))

        if year_range == (None,None):
            year_range_set = set(range(1981, this_year+1))
        else:
            year_range_set = set(range(year_range[0], year_range[1]+1))

        if hindcast:
            # SEAS5 hindcast data with 25 ensemble members
            years_hindcast_set = year_range_set.intersection(years_hindcast)
        else:
            years_hindcast_set = set()

        if forecast:
            # SEAS5 operational data with 51 ensemble members
            years_forecast_set = year_range_set.intersection(years_forecast)
        else:
            years_forecast_set = set()

        years_set = years_hindcast_set.union(years_forecast_set)
        years = range(min(years_set), max(years_set)+1)
        return years

    def _get_seas51_month(self, vname, year, month, outpath):
        """Function to retrieve seasonal surface forecasts at monthly
        resolution from ECMWF SEAS5 system in grib format.
        """

        fname = f'{vname}_{year}_{month:02}.grib'
        self.c.retrieve('seasonal-monthly-single-levels',
                        {'format': 'grib',
                         'originating_centre': 'ecmwf',
                         'system': '51',
                         'variable': self.varmap[vname],
                         'product_type': ['monthly_mean',
                                          'monthly_standard_deviation',
                                          'monthly_maximum','monthly_minimum'],
                         'year': year,
                         'month': month,
                         'leadtime_month': [1,2,3,4,5,6]}
                       ).download(os.path.join(outpath, fname))

    def download(self, vname, outpath, year_range=(None,None), months=None,
                 hindcast=False, forecast=True, overwrite=False, skip_error=False):
        """Download SEAS5 hindcast (1981-2016) or operational (2017-present)
        seasonal monthly statistics on single levels for a single variable.

        Parameters
        ----------
            vname : str
                Variable name (internal).
            outpath : str
                Output path to save files.
            year_range : (int, int), optional
                Year range to download.
            months : list, optional
                List of months to download. Defaults to full year.
            hindcast : boolean, optional
                Download hindcast data (1981-2016). Defaults to False.
            forecast : boolean, optional
                Download forecast data (2017-present). Defaults to True.
            overwrite : boolean, optional
                If True, don't check for existence of file before downloading.
                Defaults to False.
            skip_error : boolean, optional
                If True, skips any download errors via try-except.
                Defaults to True.
        """

        years = self._year_range2years(year_range, hindcast, forecast)
        if months is None:
            months = range(1,13)

        # Loop over years and months
        for year in years:
            for month in months:
                fname = f'{vname}_{year}_{month:02}.grib'
                if not overwrite and os.path.exists(os.path.join(outpath, fname)):
                    print(f'Skipping {fname} as it exists in directory.')
                else:
                    if skip_error:
                        try:
                            self._get_seas51_month(vname, year, month, outpath)
                        except:
                            print(f'*** FAILED {vname} {year}-{month:02} ***')
                    else:
                        self._get_seas51_month(vname, year, month, outpath)

    def convert(self, ds, vname, additive, lat_range=(None,None),
                lon_range=(None,None)):
        """Convert units and structure of raw files.

        Currently supported variables:
            pre - precipitation [tprate, m/s => mm]
            tmax - maximum 2m daily temperature [mx2t24, K => C]
            tmin - minimum 2m daily temperature [mn2t24, K => C]
            sst - sea surface temperature [sst, K => C]

        Parameters
        ----------
            ds : xarray Dataset or DataArray
                Dataset or DataArray with dims
                ['number','step','latitude','longitude'].
            vname : str
                Variable name (internal).
            additive : bool
                Flag additive conversions (e.g. K => C for temperature)
                which applies only to the mean not to the standard deviation.
                Contrast multiplicative conversions (like m/s => mm for precip)
                which apply to both the mean and the standard deviation.
            lat_range : (float, float), optional
                Latitude range subset to use.
            lon_range : (float, float), optional
                Longitude range subset to use.

        Returns
        -------
            ds : xarray Dataset or DataArray
                Converted Dataset or DataArray.
        """

        # Sense check that only a single month is being used
        if ds.time.ndim > 0:
            print('Dataset has more than one forecast date'
                ' - ensure file has a single forecast date only.')
            return None

        secs_per_day = 86400
        mm_per_m = 1000
        K_to_C = -273.15

        # Convert reference times and steps to effective dates of forecast
        ref_date = ds.time.values
        eff_date = ref_date + ds.step.values

        # Assign effective date to step dimension and rename
        ds = ds.drop(['surface','valid_time','time']
                    ).assign_coords({'step': eff_date}).rename({'step': 'time'})

        # Convert longitudes from 0->360 to -180->180
        ds['longitude'] = ((ds['longitude'] + 180) % 360) - 180
        ds = ds.sortby(['latitude','longitude'])

        if vname == 'pre':
            # Precipitation conversion factor from m/second to mm/day
            days_per_month = ds.time.dt.days_in_month
            ds = ds['tprate'] * days_per_month * secs_per_day * mm_per_m
        elif vname in ['tmax','tmin','sst'] and additive:
            # Temperature conversion from Kelvin to Celsius
            ds = ds[self.varmap_da[vname]] + K_to_C
        elif  vname in ['tmax','tmin','sst']:
            # No need to convert for e.g. stdev
            ds = ds[self.varmap_da[vname]]
        else:
            print(f'vname must be one of {list(self.varmap_da.keys())}')
            return None

        # Slice to lat/lon bounding box and return
        ds = ds.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))
        return ds

    def _to_monthly(self, ds):
        """Convert timestamp dimension into ['year','month'] dimensions."""
        year = ds.time.dt.year
        month = ds.time.dt.month

        # Assign new coords
        ds = ds.assign_coords(year=('time', year.data), month=('time', month.data))

        # reshape the array to (..., 'month', 'year')
        return ds.set_index(time=('year', 'month')).unstack('time')

    def proc(self, inpath, vname, month, year_range=(None, None), hindcast=False,
             forecast=True, to_monthly=True, lat_range=(None, None), lon_range=(None, None)):
        """Process multiple SEAS5 seasonal forecasts on single levels.

        Process seasonal forecast monthly statistics on single levels for a
        single month-variable. Assumes standard SEAS5 System 51 file structure
        with an internally-defined filename convention, and files in
        xarray Dataset format with dimensions [number, step, latitude, longitude],
        converts to a Dataset with dimensions [number, time, latitude, longitude],
        or [number, year, month, latitude, longitude] if to_monthly is True.

        Parameters
        ----------
            inpath : str
                Path to SEAS5 grib files by month and variable.
            vname : str
                Variable name (internal).
            month : int
                Forecast month.
            year_range : (int, int), optional
                Year range to process.
            hindcast : boolean, optional
                Use hindcast data (1981-2016). Defaults to False.
            forecast : boolean, optional
                Use forecast data (2017-present). Defaults to True.
            to_monthly : boolean, optional
                Convert datetimes to (year, month) dimensions.
            lat_range : (float, float), optional
                Latitude range subset to use to fit the model.
            lon_range : (float, float), optional
                Longitude range subset to use to fit the model.

        Returns
        -------
            ds : xarray.Dataset
                Processed Dataset.
        """

        # Generate all file paths
        years = self._year_range2years(year_range, hindcast, forecast)
        fpaths = [os.path.join(inpath, f'{vname}_{year}_{month:02}.grib')
                  for year in years]

        # Generate combined DataArray for all months for this variable
        da_mean = xr.concat([self.convert(xr.open_dataset(fpath,
                                                          filter_by_keys={'dataType': 'fcmean'},
                                                          engine='cfgrib',
                                                          backend_kwargs={'indexpath':''}),
                                          vname=vname, additive=True,
                                          lat_range=lat_range, lon_range=lon_range)
                            for fpath in fpaths], dim='time')
        da_stdev = xr.concat([self.convert(xr.open_dataset(fpath,
                                                           filter_by_keys={'dataType': 'fcstdev'},
                                                           engine='cfgrib',
                                                           backend_kwargs={'indexpath':''}),
                                           vname=vname, additive=False,
                                           lat_range=lat_range, lon_range=lon_range)
                            for fpath in fpaths], dim='time')
        return xr.Dataset({'mean': da_mean if not to_monthly else self._to_monthly(da_mean),
                           'stdev': da_stdev if not to_monthly else self._to_monthly(da_stdev)})

    def calc_clim(self, inpath, vname, year_from, year_to):
        """Calculate climatology for all months.

        Parameters
        ----------
            inpath : str
                Input path to raw download data.
            vname : str
                Variable name (internal).
            year_from : int
                Year from.
            year_to : int
                Year to.
        """
        clim = [self.proc(inpath, vname, fmonth, hindcast=True,
                          year_range=(year_from,year_to)
                          )['mean'].mean(dim=['year','number'])
                for fmonth in tqdm(range(1, 13))]
        return xr.concat(clim, dim=pd.Index(range(1, 13), name='fmonth'))

    def load_clim(self, vname, year_from, year_to):
        """Load precomputed climatology.

        Parameters
        ----------
            vname : str
                Variable name (internal).
            year_from : int
                Year from.
            year_to : int
                Year to.
        """

        fname =  f'{vname}_{year_from}_{year_to}.zarr'
        self.clim = xr.open_dataset(os.path.join(self.climpath, fname),
                                    engine='zarr')['mean'].rename(vname)
        return self.clim

    def calc_anoms(self, ds, fyear, fmonth, month):
        """Calculate forecast anomalies from forecast and climatology.

        Parameters
        ----------
            ds : xarray.Dataset
                Single month-variable's forecast as produced by self.proc().
            fyear : int
                Year of forecast.
            fmonth : int
                Month of forecast.
            month : int
                Month for which the forecast is being made.

        Returns
        -------
            ds : xarray.Dataset
                Processed Dataset.
        """

        # Calulate year (for lead times <12 months)
        lead_months = (month - fmonth) % 12
        year = fyear + (fmonth + lead_months)//12
        return (ds.sel(year=year, month=month) -
                self.clim.sel(fmonth=fmonth, month=month))


# If running the module as a whole, only download a single month's forecast
if __name__ == '__main__':
    # Always assume that cdsapi_key, vname and outpath will be passed
    cdsapi_key = sys.argv[1]
    outpath = sys.argv[2]
    vname = sys.argv[3]

    seas5 = SEAS5(cdsapi_key)
    if len(sys.argv) == 4:
        # No year or month passed
        now = dt.date.today()
        seas5.download(vname, outpath, year_range=(now.year, now.year),
                       months=[now.month], hindcast=False, forecast=True,
                       overwrite=False)
    else:
        year = sys.argv[4]
        month = sys.argv[5]
        seas5.download(vname, outpath, year_range=(int(year), int(year)),
                       months=[int(month)], hindcast=False, forecast=True,
                       overwrite=False)
