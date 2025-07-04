import argparse
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

def main(args):
    dir = args.data_dir
    if not args.skip_tsne:
        if args.model == 'val':
            filepaths = [os.path.join(dir, 'val_features_dump.pt'), os.path.join(dir, 'test_features_dump.pt')]
        elif args.model == 'best':
            filepaths = [os.path.join(dir, 'val_test_best_features_dump.pt'), os.path.join(dir, 'test_best_features_dump.pt')]
        elif args.model == 'last':
            filepaths = [os.path.join(dir, 'val_last_features_dump.pt'), os.path.join(dir, 'test_last_features_dump.pt')]
        else:
            raise ValueError(f"Unknown model {args.model}")
        features = []
        labels = []
        domains = []
        weights = []
        biases = []
        # Load
        for fp in filepaths:
            data = torch.load(fp, map_location=torch.device('cpu'))
            features.append(data['features'].numpy())  # convert to numpy
            labels.append(data['labels'].numpy())
            domains.append(data['domains'].numpy())
            weights.append(data['weights'].numpy()) # (num_classes, embed_dim)
            biases.append(data['biases'].numpy())   # (num_classes,)
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        domains = np.concatenate(domains, axis=0)
        weights = np.concatenate(weights, axis=0)
        biases = np.concatenate(biases, axis=0)
        n_classes = data['n_classes'].numpy()     # all models have the same number; use the last one

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=0, verbose=2, max_iter=args.max_iter, init='pca')
        features_2d = tsne.fit_transform(np.concatenate(features, weights, axis=0))
        weights_2d = features_2d[-n_classes,:]  
        np.delete(features_2d, [-num_classes:], axis=0)

        np.save(os.path.join(dir,f"features_2d_{args.model}.npy"), features_2d)
        np.save(os.path.join(dir,f"labels_{args.model}.npy"), labels)
        np.save(os.path.join(dir,f"domains_{args.model}.npy"), domains)
        np.save(os.path.join(dir,f"weights_{args.model}.npy"), weights)
        
    else:
        features_2d = np.load(os.path.join(dir,f"features_2d_{args.model}.npy"))
        labels = np.load(os.path.join(dir,f"labels_{args.model}.npy"))
        domains = np.load(os.path.join(dir,f"domains_{args.model}.npy"))
        weights = np.load(os.path.join(dir,f"weights_{args.model}.npy"))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    i = 0
    lls = labels
    target = "class"

    u_lls = np.unique(lls)
    for li, l in enumerate(u_lls):
        idx = (lls == l)
        axs[i].scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"{target}: {l}", alpha=0.6, s=2, marker=".")
        axs[i].scatter(weights[li, 0], weights[li, 1], label=f"weight {target}: {l}", alpha=0.6, s=4, marker="*")

    axs[i].legend()
    axs[i].set_title(f"t-SNE colored by {target}")

    i += 1
    lls = domains
    target = "domain"

    u_lls = np.unique(lls)
    for l in u_lls:
        idx = (lls == l)
        axs[i].scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"{target}: {l}", alpha=0.6, s=2, marker=".")

    axs[i].legend()
    axs[i].set_title(f"t-SNE colored by {target}")
    
    fig.suptitle(f"model: {args.model}")

    plt.savefig(os.path.join(dir, f"tsne_{args.model}.jpg"), format='jpg')
    os.startfile(os.path.abspath(os.path.join(dir, f"tsne_{args.model}.jpg")))
    #plt.show(block=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='t-sne')
    parser.add_argument('--data_dir', type=str, default='./results/CMNIST/')
    parser.add_argument('--skip_tsne', action='store_true')
    parser.add_argument('--model', type=str, default="val", choices=["val", "best", "last"])
    parser.add_argument('--max_iter', type=int, default=1000, help='for tsne; at least 250')
    parser.add_argument('--perplexity', type=int, default=30)
    args = parser.parse_args()

    main(args)
