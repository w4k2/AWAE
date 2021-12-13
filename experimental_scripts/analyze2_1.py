import numpy as np
from itertools import combinations
from tabulate import tabulate
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams
from strlearn.utils import scores_to_cummean

### E1 get end

np.set_printoptions(precision=3)

results = np.squeeze(np.load("results2_1.npy"))
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
print(results.shape)

names = os.listdir("./dd_streams")
names.sort()


bw = 8
fig, ax = plt.subplots(len(names)//2, 6,
                       figsize=(bw,bw*1.218),
                       sharex=True,
                       sharey=True)

for id ,name in enumerate(names):
    name_scores = results[id]
    print(name_scores.shape)
    for base_alg in range(3):
        for method in range(name_scores.shape[1]):
            aa = ax[id//2,base_alg+(id%2)*3]

            vw = name_scores[base_alg, method][np.newaxis,:,np.newaxis]
            vw = scores_to_cummean(vw)[0,:,0]

            aa.plot(
                vw,
                label = methods[method],
                c = colors[method],
                lw=1,
                ls=lines[method])

            if base_alg==1:
                aa.set_title('\n'.join([' '.join(name[:-4].split('_'))]+['HT']), fontsize=8)

            if base_alg==0:
                aa.set_title('GNB', fontsize=8)

            if base_alg==2:
                aa.set_title('MLP', fontsize=8)

            aa.set_ylim(0.5, 1)
            aa.spines['top'].set_visible(False)
            aa.spines['right'].set_visible(False)
            aa.grid(ls=":")
            aa.set_xticks(np.linspace(0,199,3))
            aa.set_yticks(np.linspace(.5,1,5), fontsize=8)
            aa.set_xticklabels(['','chunks',''], fontsize=7)
            aa.set_xlim(0,199)


handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=7, frameon=False)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=.5, top=.92)
plt.savefig("figures/dd_streams/%s.png" % name[:-4], dpi=150)
plt.savefig('foo.png')
plt.savefig("figures/dd_streams.eps")
plt.savefig("figures/dd_streams.png")
