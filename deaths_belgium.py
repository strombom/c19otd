
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import lognorm

countries = {
    "sweden": 'Sweden',
    "uk": 'United Kingdom'
}

start_date = "2020-01-01"
first_death = "2020-03-12"
lockdown_day_of_year = 72 # March 12
truncate_tail = 50

# Data souce (FHM) https://www.folkhalsomyndigheten.se/smittskydd-beredskap/utbrott/aktuella-utbrott/covid-19/bekraftade-fall-i-sverige/
new_deaths = {
    "sweden": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([1, 0, 1, 1, 2, 2, 1, 6, 7, 9, 8, 11, 11, 21, 22, 31, 32, 35, 38, 45, 48, 53, 70, 80, 70, 85, 90, 84, 115, 86, 90, 103, 97, 85, 91, 115, 111, 82, 86, 88, 84, 62, 77, 86, 89, 73, 75, 73, 82, 84, 78, 78, 73, 75, 84, 72, 73, 80, 60, 67, 74, 64, 61, 50, 46, 57, 48, 53, 61, 39, 54, 53, 55, 56, 43, 42, 28, 39, 40, 40, 39, 45, 40, 36, 26, 45, 38, 29, 33, 38, 34, 40, 34, 29, 33, 26, 29, 28, 32, 29, 28, 28, 20, 19, 25, 22, 20, 9, 13, 20, 17, 15, 9, 13, 6, 8, 5, 7, 4, 1, 2, 0]), np.zeros(truncate_tail, dtype=np.int))),
    "finland": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 3, 2, 2, 2, 4, 0, 2, 1, 5, 2, 0, 7, 6, 2, 6, 1, 7, 3, 5, 8, 3, 7, 8, 4, 4, 43, 8, 23, 5, 9, 4, 3, 6, 7, 5, 7, 2, 10, 10, 6, 6, 3, 5, 5, 2, 4, 4, 9, 3, 6, 4, 1, 2, 1, 3, 2, 0, 0, 1, 1, 4, 1, 0, 1, 2, 4, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), np.zeros(truncate_tail, dtype=np.int))),
    "norway": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([0, 1, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 1, 2, 2, 2, 2, 4, 2, 4, 2, 4, 10, 2, 6, 8, 1, 10, 11, 8, 4, 6, 5, 11, 13, 3, 6, 0, 12, 6, 0, 9, 6, 11, 11, 2, 0, 0, 2, 7, 2, 0, 0, 4, 0, 1, 0, 0, 4, 0, 4, 7, 4, 1, 3, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]), np.zeros(truncate_tail, dtype=np.int))),
    "denmark": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([0, 0, 0, 0, 1, 0, 3, 0, 2, 3, 4, 0, 11, 8, 2, 7, 11, 13, 7, 5, 13, 14, 19, 16, 22, 18, 8, 16, 15, 19, 10, 13, 13, 12, 14, 10, 12, 15, 10, 9, 9, 6, 14, 10, 9, 15, 4, 5, 7, 9, 9, 10, 11, 8, 9, 7, 3, 6, 6, 4, 3, 5, 4, 6, 4, 0, 5, 4, 1, 3, 3, 7, 0, 0, 1, 1, 0, 2, 3, 0, 3, 3, 2, 4, 0, 2, 4, 1, 2, 4, 0, 0, 0, 1, 3, 0, 1, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0]), np.zeros(truncate_tail, dtype=np.int))),
    "belgium": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([1, 3, 5, 6, 11, 12, 22, 28, 31, 39, 42, 77, 74, 97, 101, 124, 134, 139, 170, 187, 249, 210, 231, 273, 233, 273, 303, 317, 271, 318, 288, 343, 280, 275, 275, 252, 204, 209, 212, 199, 202, 192, 175, 147, 155, 176, 127, 115, 98, 84, 98, 73, 96, 98, 82, 75, 78, 70, 68, 73, 62, 48, 54, 46, 37, 28, 35, 35, 30, 32, 29, 37, 24, 35, 24, 40, 24, 22, 13, 24, 19, 22, 20, 20, 14, 12, 11, 11, 9, 12, 12, 7, 10, 5, 3, 8, 7, 7, 5, 4, 3, 9, 12, 4, 3, 5, 2, 6, 5, 2, 0, 2, 7, 4, 6, 0, 0, 3, 2, 2, 3, 1]), np.zeros(truncate_tail, dtype=np.int))),
    "uk": np.concatenate((np.zeros(lockdown_day_of_year - 1, dtype=np.int), np.array([0, 2, 1, 18, 15, 22, 16, 34, 43, 36, 56, 35, 74, 149, 186, 183, 284, 294, 214, 374, 382, 670, 652, 714, 760, 644, 568, 1038, 1034, 1103, 1152, 839, 686, 744, 1044, 842, 1029, 935, 1115, 498, 559, 1172, 837, 727, 1005, 843, 420, 338, 909, 795, 674, 739, 621, 315, 288, 693, 649, 539, 626, 346, 268, 210, 627, 494, 428, 384, 468, 170, 160, 545, 363, 338, 351, 282, 118, 121, 134, 412, 377, 324, 215, 113, 556, 324, 359, 176, 357, 204, 77, 55, 286, 245, 151, 202, 181, 36, 38, 233, 184, 135, 173, 128, 43, 15, 280, 154, 149, 184, 100, 36, 25, 155, 176, 89, 136, 67, 22, 16, 155, 126, 85, 48]), np.zeros(truncate_tail, dtype=np.int)))
}

n_samples = len(new_deaths["sweden"])
t = np.linspace(0, n_samples-1, n_samples)

# Novus https://novus.se/novus-coronastatus/
hypochondriacs = 0.5
novus = np.interp(x=t, 
    xp=[0,       31,   50,   60,  72,   94,  117], 
    fp=[0.004, 0.01, 0.04, 0.07, 0.1, 0.13, 0.15]
    )[:n_samples - truncate_tail * 2] * 100000 * hypochondriacs

for country in new_deaths:
    for i in range(4):
        new_deaths[country] = np.convolve(new_deaths[country], [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006], 'same') # Gaussian filter

x_dates = pd.date_range(start_date, periods=n_samples, freq='D')

# Plot
colors = {
    "sweden": 'g',
    "denmark": 'y',
    "finland": 'b',
    "norway": 'r',
    "belgium": 'r',
    "uk": 'r'
}

decline_start = 115
decline_end = 185

area_se = new_deaths["sweden"] / max(new_deaths["sweden"])
area_be = new_deaths["uk"] / max(new_deaths["uk"])
area_decline_se = area_se[decline_start:decline_end]
area_decline_be = area_be[decline_start:decline_end]

area_decline_diff = area_decline_se - area_decline_be
area = (sum(area_decline_se) - sum(area_decline_be)) / sum(area_se)

print("se area tot", sum(area_se))
print("no area tot", sum(area_be))
print("diff area tot", sum(area_se) - sum(area_be))
print("diff decline", sum(area_decline_diff))
print("separt", area)


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
for country in countries:
    ax.plot(x_dates, new_deaths[country] / max(new_deaths[country]), colors[country], alpha=0.8, lw=1.5, label=countries[country])

ax.fill_between(x_dates[decline_start:decline_end], area_decline_se, area_decline_be, 'g', alpha=0.5)

ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
ax.set_title("Covid-19 diseased relative to peak")
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.0)
ax.text(0.54, 0.5, "%0.1f %%" % (area*100), color='w', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.show()
