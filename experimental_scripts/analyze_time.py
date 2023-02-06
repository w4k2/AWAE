import numpy as np
import matplotlib.pyplot as plt


results = np.load("results/experiment_time/times.npy")
results = np.mean(results, axis=0)
results = results[[0, 1, 2, 3, 4, 6, 7, 5]]
print(results)

methods = ["SEA", "AWE", "AUE", "NSE", "OALE", "KUE", "ROSE", "AWAE"]
colors = ["black", "blue", "blue", "green", "green", "orange", "orange", "red"]

index = np.arange(len(methods))
bar_width = 0.4

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# for id, result in enumerate(results):
#     plt.bar(index, result, bar_width, color=colors[id])
plt.bar(np.arange(len(methods)), results, color=colors, width=bar_width)
plt.yscale("log")
plt.ylim( (10**-1,10**2) )
plt.ylabel("Processing time [s]", size=20)
plt.yticks(size=18)
plt.xticks(index, methods, size=18)
plt.xlabel("Method", size=20)
plt.tight_layout()
plt.grid(ls=":", c=(.7, .7, .7))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("bar.png", dpi=200)
plt.savefig("bar.eps")
