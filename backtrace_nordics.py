
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import lognorm


start_date = "2020-01-01"
first_death = "2020-03-12"
lockdown_day_of_year = 72 # March 12


# Data source (time to death) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7074197/
otd_samples = [20, 15, 6, 10, 14, 41, 12, 30, 7, 19, 13, 10, 11, 19, 16, 17, 24, 8, 12, 8, 6, 12, 13, 16, 11, 12, 21, 10, 11, 11, 14, 32, 18, 13]
otd_shape, otd_loc, otd_scale = lognorm.fit(otd_samples)
eto_samples = [5, 4, 6, 5, 29, 4, 6, 8, 7, 21, 1, 7, 8, 8, 9, 3, 8, 4, 5, 3, 4, 11, 5, 9, 4, 3, 6, 5, 6, 11, 12, 4, 8, 14, 3, 7, 28, 5, 6, 9, 3, 2, 2, 14, 32, 7, 2, 15, 8, 9, 3, 2, 2, 14, 32, 7, 2, 15, 8]
eto_shape, eto_loc, eto_scale = lognorm.fit(eto_samples)
truncate_tail = 50

def sample_exposure_to_death():
    exposure_to_onset = lognorm.rvs(s=eto_shape, loc=eto_loc, scale=eto_scale)
    onset_to_death = math.inf
    while onset_to_death > truncate_tail:
        onset_to_death = lognorm.rvs(s=otd_shape, loc=otd_loc, scale=otd_scale)
    return int(exposure_to_onset + onset_to_death)


# Data souce (FHM) https://www.folkhalsomyndigheten.se/smittskydd-beredskap/utbrott/aktuella-utbrott/covid-19/bekraftade-fall-i-sverige/
new_deaths = {
    "sweden": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([1, 0, 1, 1, 2, 2, 1, 6, 7, 9, 8, 11, 11, 21, 22, 31, 32, 35, 38, 45, 48, 53, 70, 80, 70, 85, 90, 84, 115, 86, 90, 103, 97, 85, 91, 115, 111, 82, 86, 88, 84, 62, 77, 86, 89, 73, 75, 73, 82, 84, 78, 78, 73, 75, 84, 72, 73, 80, 60, 67, 74, 64, 61, 50, 46, 57, 48, 53, 61, 39, 54, 53, 55, 56, 43, 42, 28, 39, 40, 40, 39, 45, 40, 36, 26, 45, 38, 29, 33, 38, 34, 40, 34, 29, 33, 26, 29, 28, 32, 29, 28, 28, 20, 19, 25, 22, 20, 9, 13, 20, 17, 15, 9, 13, 6, 8, 5, 7, 4, 1, 2, 0]), np.zeros(truncate_tail, dtype=np.int))),
    "finland": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 3, 2, 2, 2, 4, 0, 2, 1, 5, 2, 0, 7, 6, 2, 6, 1, 7, 3, 5, 8, 3, 7, 8, 4, 4, 43, 8, 23, 5, 9, 4, 3, 6, 7, 5, 7, 2, 10, 10, 6, 6, 3, 5, 5, 2, 4, 4, 9, 3, 6, 4, 1, 2, 1, 3, 2, 0, 0, 1, 1, 4, 1, 0, 1, 2, 4, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), np.zeros(truncate_tail, dtype=np.int))),
    "norway": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([0, 1, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 1, 2, 2, 2, 2, 4, 2, 4, 2, 4, 10, 2, 6, 8, 1, 10, 11, 8, 4, 6, 5, 11, 13, 3, 6, 0, 12, 6, 0, 9, 6, 11, 11, 2, 0, 0, 2, 7, 2, 0, 0, 4, 0, 1, 0, 0, 4, 0, 4, 7, 4, 1, 3, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]), np.zeros(truncate_tail, dtype=np.int))),
    "denmark": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([0, 0, 0, 0, 1, 0, 3, 0, 2, 3, 4, 0, 11, 8, 2, 7, 11, 13, 7, 5, 13, 14, 19, 16, 22, 18, 8, 16, 15, 19, 10, 13, 13, 12, 14, 10, 12, 15, 10, 9, 9, 6, 14, 10, 9, 15, 4, 5, 7, 9, 9, 10, 11, 8, 9, 7, 3, 6, 6, 4, 3, 5, 4, 6, 4, 0, 5, 4, 1, 3, 3, 7, 0, 0, 1, 1, 0, 2, 3, 0, 3, 3, 2, 4, 0, 2, 4, 1, 2, 4, 0, 0, 0, 1, 3, 0, 1, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0]), np.zeros(truncate_tail, dtype=np.int)))
}
n_samples = len(new_deaths["sweden"])
t = np.linspace(0, n_samples, n_samples)


# Estimate day of infection
infected = {}
for country in new_deaths:
    infected[country] = np.zeros(n_samples, dtype=np.int)
    for day in range(n_samples):
        for dead in range(new_deaths[country][day]):
            exposure_day = max(0, day - sample_exposure_to_death())
            infected[country][exposure_day] += 1.0

capita = {
    "sweden": 100/10.3,
    "finland": 100/5.5,
    "norway": 100/5.5,
    "denmark": 100/5.5
}
for country in new_deaths:
    infected[country] *= int(capita[country])

# Plot
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
x_dates = pd.date_range(start_date, periods=n_samples, freq='D')
ax.plot(x_dates, infected["sweden"], 'g', alpha=0.8, lw=1.5, label='Infected Sweden')
ax.plot(x_dates, infected["finland"], 'b', alpha=0.8, lw=1.5, label='Infected Finland')
ax.plot(x_dates, infected["norway"], 'r', alpha=0.8, lw=1.5, label='Infected Norway')
ax.plot(x_dates, infected["denmark"], 'y', alpha=0.8, lw=1.5, label='Infected Denmark')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()

