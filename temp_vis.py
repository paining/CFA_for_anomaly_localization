import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

with open("backup.pkl", "rb") as f:
    loaded = pickle.load(f)

manifold_features = loaded["manifold_features"]
labels = loaded["labels"]
labels = np.array(labels).squeeze(-1)
fig, subplot = plt.subplots(1,1)

color_list = ["red", "blue", "darkgreen", "orange", "puple", "skyblue"]
for l in range(labels.min(), labels.max()+1):
    idx = labels == l
    subplot.scatter(
        manifold_features[idx,0], 
        manifold_features[idx,1], 
        c=color_list[l]
    )

plt.legend([str(i) for i in range(labels.min(), labels.max()+1)])
plt.show()