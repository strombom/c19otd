
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import lognorm

countries = {
    "sweden": 'Sweden',
    #"denmark": 'Denmark',
    "norway": 'Norway'
    #"finland": 'Finland'
}

population = {
    "sweden": 10340000,
    "denmark": 5820000,
    "norway": 5370000,
    "finland": 5530000
}

age_structure = { # 0-14y, 15-64y, +65y
    "sweden": (0.175, 0.625, 0.199),
    "denmark": (0.165, 0.638, 0.197),
    "norway": (0.178, 0.654, 0.168),
    "finland": (0.164, 0.624, 0.212)
}

ifr_by_age = []

start_date = "2020-01-01"
first_death = "2020-03-12"
lockdown_day_of_year = 72 # March 12

# Assuming infected population is young at start and the elderly population is better protected towards the end
ifr_start, ifr_lockdown, ifr_end = 0.006, 0.008, 0.004

# Data source (time to death) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7074197/
otd_samples = [20, 15, 6, 10, 14, 41, 12, 30, 7, 19, 13, 10, 11, 19, 16, 17, 24, 8, 12, 8, 6, 12, 13, 16, 11, 12, 21, 10, 11, 11, 14, 32, 18, 13]
otd_shape, otd_loc, otd_scale = lognorm.fit(otd_samples)
eto_samples = [5, 4, 6, 5, 29, 4, 6, 8, 7, 21, 1, 7, 8, 8, 9, 3, 8, 4, 5, 3, 4, 11, 5, 9, 4, 3, 6, 5, 6, 11, 12, 4, 8, 14, 3, 7, 28, 5, 6, 9, 3, 2, 2, 14, 32, 7, 2, 15, 8, 9, 3, 2, 2, 14, 32, 7, 2, 15, 8]
eto_shape, eto_loc, eto_scale = lognorm.fit(eto_samples)
truncate_tail = 40

# Data souce (FHM) https://www.folkhalsomyndigheten.se/smittskydd-beredskap/utbrott/aktuella-utbrott/covid-19/bekraftade-fall-i-sverige/
new_deaths = {
    "sweden": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([1, 0, 1, 1, 2, 2, 1, 6, 7, 9, 8, 11, 11, 21, 22, 31, 32, 35, 38, 45, 48, 53, 70, 80, 70, 85, 90, 84, 115, 86, 90, 103, 97, 85, 91, 115, 111, 82, 86, 88, 84, 62, 77, 86, 89, 73, 75, 73, 82, 84, 78, 78, 73, 75, 84, 72, 73, 80, 60, 67, 74, 64, 61, 50, 46, 57, 48, 53, 61, 39, 54, 53, 55, 56, 43, 42, 28, 39, 40, 40, 39, 45, 40, 36, 26, 45, 38, 29, 33, 38, 34, 40, 34, 29, 33, 26, 29, 28, 32, 29, 28, 28, 20, 19, 25, 22, 20, 9, 13, 20, 17, 15, 9, 13, 6, 8, 5, 7, 4, 1, 2, 0]), np.zeros(truncate_tail, dtype=np.int))),
    "finland": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 3, 2, 2, 2, 4, 0, 2, 1, 5, 2, 0, 7, 6, 2, 6, 1, 7, 3, 5, 8, 3, 7, 8, 4, 4, 43, 8, 23, 5, 9, 4, 3, 6, 7, 5, 7, 2, 10, 10, 6, 6, 3, 5, 5, 2, 4, 4, 9, 3, 6, 4, 1, 2, 1, 3, 2, 0, 0, 1, 1, 4, 1, 0, 1, 2, 4, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), np.zeros(truncate_tail, dtype=np.int))),
    "norway": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([0, 1, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 1, 2, 2, 2, 2, 4, 2, 4, 2, 4, 10, 2, 6, 8, 1, 10, 11, 8, 4, 6, 5, 11, 13, 3, 6, 0, 12, 6, 0, 9, 6, 11, 11, 2, 0, 0, 2, 7, 2, 0, 0, 4, 0, 1, 0, 0, 4, 0, 4, 7, 4, 1, 3, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]), np.zeros(truncate_tail, dtype=np.int))),
    "denmark": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([0, 0, 0, 0, 1, 0, 3, 0, 2, 3, 4, 0, 11, 8, 2, 7, 11, 13, 7, 5, 13, 14, 19, 16, 22, 18, 8, 16, 15, 19, 10, 13, 13, 12, 14, 10, 12, 15, 10, 9, 9, 6, 14, 10, 9, 15, 4, 5, 7, 9, 9, 10, 11, 8, 9, 7, 3, 6, 6, 4, 3, 5, 4, 6, 4, 0, 5, 4, 1, 3, 3, 7, 0, 0, 1, 1, 0, 2, 3, 0, 3, 3, 2, 4, 0, 2, 4, 1, 2, 4, 0, 0, 0, 1, 3, 0, 1, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0]), np.zeros(truncate_tail, dtype=np.int)))
}

n_samples = len(new_deaths["sweden"])
t = np.linspace(0, n_samples-1, n_samples)

# Novus https://novus.se/novus-coronastatus/
hypochondriacs = 0.5
novus = np.interp(x=t, 
    xp=[0,       31,   50,   60,  72,   94,  117], 
    fp=[0.004, 0.01, 0.04, 0.07, 0.1, 0.13, 0.15]
    )[:n_samples - truncate_tail * 2] * 100000 * hypochondriacs


def sample_exposure_to_death():
    exposure_to_onset = lognorm.rvs(s=eto_shape, loc=eto_loc, scale=eto_scale)
    onset_to_death = math.inf
    while onset_to_death > truncate_tail:
        onset_to_death = lognorm.rvs(s=otd_shape, loc=otd_loc, scale=otd_scale)
    return int(exposure_to_onset + onset_to_death)

# Set IFR
ifr = np.interp(x=t, xp=[0, lockdown_day_of_year, n_samples-1], fp=[ifr_start, ifr_lockdown, ifr_end])
n_infected_per_death = 1 / ifr

# Estimate day of infection
n_montecarlo = 10
infected = {}
for country in new_deaths:
    infected[country] = np.zeros(n_samples, dtype=np.float)
    for day in range(n_samples):
        for dead in range(new_deaths[country][day]):
            for i in range(n_montecarlo):
                exposure_day = max(0, day - sample_exposure_to_death())
                infected[country][exposure_day] += n_infected_per_death[exposure_day]
    infected[country] /= n_montecarlo

for country in new_deaths:
    infected[country] = 100000 * infected[country] / population[country]
    infected[country] = np.convolve(infected[country], [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006], 'same') # Gaussian filter
    infected[country] = infected[country][:n_samples - truncate_tail * 2]

x_dates = pd.date_range(start_date, periods=n_samples - truncate_tail * 2, freq='D')

# Plot
colors = {
    "sweden": 'g',
    "denmark": 'y',
    "finland": 'b',
    "norway": 'r'
}

decline_start = 85
decline_se = infected["sweden"] / max(infected["sweden"])
decline_no = infected["norway"] / max(infected["norway"])
decline_se[0:decline_start] = decline_se[decline_start]
decline_no[0:decline_start] = decline_se[decline_start]

decline_diff =  sum(decline_se)


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
for country in countries:
    ax.plot(x_dates, infected[country] / max(infected[country]), colors[country], alpha=0.8, lw=1.5, label=countries[country])

ax.fill_between(x_dates, decline_se, decline_no, 'g', alpha=0.5)

ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
ax.set_title("Newly infected per day, per 100 000 inhabitants")
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
"""
ax = fig.add_subplot(212, facecolor='#dddddd', axisbelow=True)
for country in countries:
    ax.plot(x_dates, np.cumsum(infected[country]), colors[country], alpha=0.8, lw=1.5, label=countries[country])
ax.plot(x_dates, novus, 'r--', alpha=0.8, lw=1.0, label="Novus")
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
ax.set_title("Cumulative infected, per 100 000 inhabitants")
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
"""
plt.show()

