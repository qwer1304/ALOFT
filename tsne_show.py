import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy
import os

dir = './results/CMNIST/'
filepath = os.path.join(dir, 'val_features_dump.pt')
# Load
data = torch.load(filepath, map_location=torch.device('cpu'))
features = data['features'].numpy()  # convert to numpy
labels = data['labels'].numpy()
domains = data['domains'].numpy()
print(domains)
exit()

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=0, verbose=2, max_iter=300, init='pca')
features_2d = tsne.fit_transform(features)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

i = 0
lls = labels
num_lls = len(set(lls))
target = "class"
for l in range(num_lls):
    idx = (lls == l)
    axs[i].scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"Class {l}", alpha=0.6, s=2, marker=".")

axs[i].legend()
axs[i].set_title(f"t-SNE colored by {target}")

i += 1
lls = domains
num_lls = len(set(lls))
target = "domain"
for l in range(num_lls):
    idx = (lls == l)
    axs[i].scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"{target}: {l}", alpha=0.6, s=2, marker=".")

axs[i].legend()
axs[i].set_title(f"t-SNE colored by {target}")

plt.show(block=True)