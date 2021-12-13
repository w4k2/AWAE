import numpy as np
from itertools import combinations
from tabulate import tabulate
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
from scipy.ndimage import gaussian_filter1d
from strlearn.utils import scores_to_cummean
from scipy.interpolate import interp1d


from matplotlib import rcParams

### E1 get end

np.set_printoptions(precision=3)
methods = ["SEA", "AWE", "AUE", "NSE", "OALE", "(d) WAE", "(o) WAE"]
colors = ["black", "blue", "blue", "green", "green", "red", "red"]
lines = ["-", "-", "--", "-", "--", "-", "--"]
base = ["GNB", "HT", "MLP"]
"""
# Dimensions are
# 0 - stream
# 1 - base_estimator
# 2 - methods
# 3 - chunk
"""

names = os.listdir("./ss_streams")
names.sort()

bw = 8
fig, ax = plt.subplots(len(names), 3,
                       figsize=(bw//2,bw*1.218),
                       sharex=True,
                       sharey=True)


for id ,name in enumerate(names):
    name_scores = np.squeeze(np.load("results/ex2_2/%s" % name))
    print(name_scores.shape)
    for axis in range(3):
        for method in range(name_scores.shape[1]):
            aa = ax[id,axis]

            vw = name_scores[axis, method][np.newaxis,:,np.newaxis]
            vw = scores_to_cummean(vw)[0,:,0]

            f = interp1d(np.linspace(0,1,vw.shape[0]),
                                     vw)
            vw = f(np.linspace(0,1,20))


            aa.plot(vw, label=methods[method], c=colors[method], ls=lines[method], lw=1)
            aa.set_ylim(0.5, 1)

            if axis==1:
                aa.set_title('\n'.join([' '.join(name[:-4].split('_'))]+['HT']), fontsize=8)

            if axis==0:
                aa.set_title('GNB', fontsize=8)

            if axis==2:
                aa.set_title('MLP', fontsize=8)



            aa.spines['top'].set_visible(False)
            aa.spines['right'].set_visible(False)
            aa.grid(ls=":")
            aa.set_xticks(np.linspace(0,20,3))
            aa.set_yticks(np.linspace(.5,1,3), fontsize=7)
            aa.set_xticklabels(['','chunks',''], fontsize=7)
            aa.set_yticklabels([
                '.5',
                '.75',
                '1.'
            ],fontsize=7)
            aa.set_xlim(0,19)

handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
           fontsize=8)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=.75, top=.92)


plt.savefig('foo.png')
plt.savefig("figures/ss_streams.eps")
plt.savefig("figures/ss_streams.png")
