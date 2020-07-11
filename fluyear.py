
import numpy as np
import matplotlib.pyplot as plt

years = ["2000-01","2001-02","2002-03","2003-04","2004-05","2005-06","2006-07","2007-08","2008-09","2009-10","2010-11","2011-12","2012-13","2013-14","2014-15","2015-16","2016-17","2017-18","2018-19","2019-20"]
nums = np.array([74965,76299,76511,73076,75154,71648,73779,72630,72834,69751,70590,71185,70840,66564,69248,65459,67000,66392,61990,66841])

n_samples = len(nums)
x = np.linspace(0, n_samples-1, n_samples)
p = np.polyfit(x[0:-1], nums[0:-1], 3)
fit = np.polyval(p, x)

extra = np.copy(nums)
diff = nums[n_samples - 1] - fit[n_samples - 1]
nums[n_samples - 1] = nums[n_samples - 1] - diff
extra[n_samples - 1] = nums[n_samples - 1] + diff

print(diff)

fig = plt.figure()
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
rect1 = ax.bar(years, extra, alpha=0.8, width=0.5, color="r")
rect2 = ax.bar(years, nums, alpha=0.8, width=0.5, color='c')


ax.plot(x, fit, 'y', alpha=0.8, lw=1.5, label='Best fit')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rect1)
autolabel(rect2)

plt.show()
