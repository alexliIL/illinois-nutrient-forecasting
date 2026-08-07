"""
Fetch site coordinates (USGS) and daily weather (Daymet) for the 8 Illinois rivers.
Writes data/site_coords.csv and data/weather_daymet.csv
"""
from dataretrieval import nwis
import pydaymet as daymet
import pandas as pd

# The 8 gage IDs — note leading zeros (CSV stored them as ints)
sites = ['03339000','03346500','03381495','05446500',
         '05447500','05586300','05595000','05599490']

# --- get coordinates ---
info, _ = nwis.get_info(sites=sites)
coords = info[['site_no', 'dec_lat_va', 'dec_long_va']].copy()
coords.to_csv('data/site_coords.csv', index=False)
print("Coordinates:")
print(coords)

# --- pull Daymet weather per site ---
frames = []
for _, r in coords.iterrows():
    clm = daymet.get_bycoords(
        (r.dec_long_va, r.dec_lat_va),
        ("2015-10-01", "2021-09-30"),
        variables=["prcp", "tmax", "tmin"],
    )
    clm = clm.reset_index()
    clm['site'] = int(r.site_no)   # strip leading zero to match M1's integer sites
    frames.append(clm)

weather = pd.concat(frames, ignore_index=True)
weather.to_csv('data/weather_daymet.csv', index=False)
print("\nWeather sample:")
print(weather.head())
print("\nColumns:", list(weather.columns))