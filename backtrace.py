
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

def sample_exposure_to_death():
    exposure_to_onset = lognorm.rvs(s=eto_shape, loc=eto_loc, scale=eto_scale)
    onset_to_death = math.inf
    truncate = 50
    while onset_to_death > truncate:
        onset_to_death = lognorm.rvs(s=otd_shape, loc=otd_loc, scale=otd_scale)
    return int(exposure_to_onset + onset_to_death)


# Data souce (FHM) https://www.folkhalsomyndigheten.se/smittskydd-beredskap/utbrott/aktuella-utbrott/covid-19/bekraftade-fall-i-sverige/
new_deaths = np.concatenate((np.zeros(71, dtype=np.int), np.array([1, 0, 1, 1, 2, 2, 1, 6, 7, 9, 8, 11, 11, 21, 22, 31, 32, 35, 38, 45, 48, 53, 70, 80, 70, 85, 90, 84, 115, 86, 90, 103, 97, 85, 91, 115, 111, 82, 86, 88, 84, 62, 77, 86, 89, 73, 75, 73, 82, 84, 78, 78, 73, 75, 84, 72, 73, 80, 60, 67, 74, 64, 61, 50, 46, 57, 48, 53, 61, 39, 54, 53, 55, 56, 43, 42, 28, 39, 40, 40, 39, 45, 40, 36, 26, 45, 38, 29, 33, 38, 34, 40, 34, 29, 33, 26, 29, 28, 32, 29, 28, 28, 20, 19, 25, 22, 20, 9, 13, 20, 17, 15, 9, 13, 6, 8, 5, 7, 4, 1, 2, 0]), np.zeros(50, dtype=np.int)))
n_samples = len(new_deaths)
t = np.linspace(0, n_samples, n_samples)


# Estimate day of infection
infected = np.zeros(n_samples, dtype=np.int)
for day in range(n_samples):
    for dead in range(new_deaths[day]):
        exposure_day = max(0, day - sample_exposure_to_death())
        infected[exposure_day] += 1


# Estimate day of death
simulated_no_lockdown = np.zeros(n_samples, dtype=np.int)
simulated_lockdown = np.zeros(n_samples, dtype=np.int)
for day in range(n_samples):
    for dead in range(infected[day]):
        death_day = min(n_samples - 1, day + sample_exposure_to_death())
        simulated_no_lockdown[death_day] += 1

        if day < lockdown_day_of_year:
            death_day = min(n_samples - 1, day + sample_exposure_to_death())
            simulated_lockdown[death_day] += 1

print("No lockdown", simulated_no_lockdown.sum())
print("Lockdown", simulated_lockdown.sum())

# Plot
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
x_dates = pd.date_range(start_date, periods=n_samples, freq='D')
ax.plot(x_dates, new_deaths, 'g', alpha=0.3, lw=1, label='Deaths (FHM)')
ax.plot(x_dates, infected, 'b', alpha=0.3, lw=1, label='Infected (Sim)')
ax.plot(x_dates, simulated_no_lockdown, 'r', alpha=0.8, lw=1.5, label='No lockdown (Sim)')
ax.plot(x_dates, simulated_lockdown, 'y', alpha=0.8, lw=1.5, label='Lockdown (Sim)')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()

