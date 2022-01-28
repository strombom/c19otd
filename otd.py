
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import lognorm

# Data source (time to death) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7074197/
otd_samples = [20, 15, 6, 10, 14, 41, 12, 30, 7, 19, 13, 10, 11, 19, 16, 17, 24, 8, 12, 8, 6, 12, 13, 16, 11, 12, 21, 10, 11, 11, 14, 32, 18, 13]
otd_shape, otd_loc, otd_scale = lognorm.fit(otd_samples)
eto_samples = [5, 4, 6, 5, 29, 4, 6, 8, 7, 21, 1, 7, 8, 8, 9, 3, 8, 4, 5, 3, 4, 11, 5, 9, 4, 3, 6, 5, 6, 11, 12, 4, 8, 14, 3, 7, 28, 5, 6, 9, 3, 2, 2, 14, 32, 7, 2, 15, 8, 9, 3, 2, 2, 14, 32, 7, 2, 15, 8]
eto_shape, eto_loc, eto_scale = lognorm.fit(eto_samples)
x = np.linspace(0,40,100)

def sample_exposure_to_death():
    exposure_to_onset = lognorm.rvs(s=eto_shape, loc=eto_loc, scale=eto_scale)
    onset_to_death = 1000
    truncate = 40
    while onset_to_death > truncate:
        onset_to_death = lognorm.rvs(s=otd_shape, loc=otd_loc, scale=otd_scale)
    return exposure_to_onset + onset_to_death

def three_weeks_infection_to_death():
    return 7 * 3


n_days = 90
time_of_first_death = 25
lockdown = 24

# Total population, N.
N = 1000000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta_0 = 0.5
beta_soft = 0.1
beta_hard = 0.05
gamma = 1./7

# A grid of time points (in days)
t = np.linspace(0, n_days, n_days)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta_0, gamma))
S_no, I_no, R_no = ret.T

ret_soft = odeint(deriv, (S_no[lockdown], I_no[lockdown], R_no[lockdown]), t[lockdown:], args=(N, beta_soft, gamma))
S_soft, I_soft, R_soft = ret_soft.T
S_soft = np.concatenate((S_no[0:lockdown], S_soft))
I_soft = np.concatenate((I_no[0:lockdown], I_soft))
R_soft = np.concatenate((R_no[0:lockdown], R_soft))

ret_hard = odeint(deriv, (S_no[lockdown], I_no[lockdown], R_no[lockdown]), t[lockdown:], args=(N, beta_hard, gamma))
S_hard, I_hard, R_hard = ret_hard.T
S_hard = np.concatenate((S_no[0:lockdown], S_hard))
I_hard = np.concatenate((I_no[0:lockdown], I_hard))
R_hard = np.concatenate((R_no[0:lockdown], R_hard))

IFR = 0.005

previously_susceptible_no = N
previously_susceptible_soft = N
previously_susceptible_hard = N

deaths_no = np.zeros(n_days)
deaths_soft = np.zeros(n_days)
deaths_hard = np.zeros(n_days)
for day_idx in range(n_days):
    newly_infected_no = int(round(previously_susceptible_no - S_no[day_idx]))
    previously_susceptible_no = S_no[day_idx]
    newly_infected_soft = int(round(previously_susceptible_soft - S_soft[day_idx]))
    previously_susceptible_soft = S_soft[day_idx]
    newly_infected_hard = int(round(previously_susceptible_hard - S_hard[day_idx]))
    previously_susceptible_hard = S_hard[day_idx]

    for inf_idx in range(newly_infected_no):
        if np.random.rand() <= IFR:
            time_of_death = int(round(day_idx + sample_exposure_to_death()))
            if time_of_death < n_days:
                deaths_no[time_of_death] += 1

    for inf_idx in range(newly_infected_soft):
        if np.random.rand() <= IFR:
            time_of_death = int(round(day_idx + sample_exposure_to_death()))
            if time_of_death < n_days:
                deaths_soft[time_of_death] += 1

    for inf_idx in range(newly_infected_hard):
        if np.random.rand() <= IFR:
            time_of_death = int(round(day_idx + sample_exposure_to_death()))
            if time_of_death < n_days:
                deaths_hard[time_of_death] += 1
            
deaths_no_cum = np.cumsum(deaths_no)
deaths_soft_cum = np.cumsum(deaths_soft)
deaths_hard_cum = np.cumsum(deaths_hard)


# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(211, facecolor='#dddddd', axisbelow=True)
#ax.plot(t, S_no/N, 'b', alpha=0.5, lw=2, label='Susceptible')
#ax.plot(t, I_no, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, I_soft, 'g', alpha=0.5, lw=2, label='Infected soft lockdown')
ax.plot(t, I_hard, 'b', alpha=0.5, lw=2, label='Infected hard lockdown')
#ax.plot(t, R_no/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Number')
#ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)

ax2 = fig.add_subplot(212, facecolor='#dddddd', axisbelow=True)
#ax2.plot(t, deaths_no*10, 'r', alpha=0.5, lw=2, label='Deaths no lockdown')
ax2.plot(t, deaths_soft*10, 'g', alpha=0.5, lw=2, label='Deaths soft lockdown')
ax2.plot(t, deaths_hard*10, 'b', alpha=0.5, lw=2, label='Deaths hard lockdown')
#ax2.plot(t, deaths_3w*10, 'g', alpha=0.5, lw=2, label='Deaths 3w fixed')
#ax2.plot(t, deaths_no_cum*10, 'g', alpha=0.5, lw=2, label='Deaths lognorm cum')
#ax2.plot(t, deaths_soft_cum*10, 'r', alpha=0.5, lw=2, label='Deaths no lockdown cum')
#ax2.plot(t, deaths_hard_cum*10, 'g', alpha=0.5, lw=2, label='Deaths 3w fixed cum')
legend = ax2.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)

plt.show()

#03-19 1
#04-19 1600







