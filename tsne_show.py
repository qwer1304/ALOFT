import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
from tqdm.auto import tqdm

def main(args):
    dir = args.data_dir
    if not args.skip_embedding:
        print('Importing ... ', end="")
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        from umap import UMAP
        print('Done!')

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
        # Load
        for fp in tqdm(filepaths, leave=False, total=len(filepaths)):
            data = torch.load(fp, map_location=torch.device('cpu'))
            features.append(data['features'].numpy())  # convert to numpy
            labels.append(data['labels'].numpy())
            domains.append(data['domains'].numpy())
        features  = np.concatenate(features, axis=0)
        labels    = np.concatenate(labels, axis=0)
        domains   = np.concatenate(domains, axis=0)
        # model parameters
        weights   = data['head_weights'].detach().numpy() # (num_classes, embed_dim)
        biases    = data['head_bias'].detach().numpy()    # (num_classes,)
        n_classes = data['n_classes']                     # int
        
        if args.method == 'tsne':
            tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=0, verbose=2, max_iter=args.max_iter, init='pca')
            features_2d = tsne.fit_transform(np.concatenate([features, weights], axis=0))

        elif args.method == 'pca':
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(np.concatenate([features, weights], axis=0))
       
        elif args.method == 'umap':
            umap = UMAP(n_components=2, n_neighbors=args.n_neighbors, min_dist=args.min_dist, verbose=True)
            features_2d = umap.fit_transform(np.concatenate([features, weights], axis=0))
           
        else:
            raise ValueError(f"Unknown method {args.method}")
            
        weights_2d = features_2d[-n_classes:,:]  
        features_2d = features_2d[:-n_classes,:]

        print('Saving embeddings ... ', end="")
        np.save(os.path.join(dir,f"features_2d_{args.method}_{args.model}.npy"), features_2d)
        np.save(os.path.join(dir,f"weights_2d_{args.model}.npy"), weights_2d)
        np.save(os.path.join(dir,f"labels_{args.model}.npy"), labels)
        np.save(os.path.join(dir,f"domains_{args.model}.npy"), domains)
        print('Done!')

    else:
        print('Loading embeddings ... ', end="")
        features_2d = np.load(os.path.join(dir,f"features_2d_{args.method}_{args.model}.npy"))
        weights_2d = np.load(os.path.join(dir,f"weights_2d_{args.model}.npy"))
        labels = np.load(os.path.join(dir,f"labels_{args.model}.npy"))
        domains = np.load(os.path.join(dir,f"domains_{args.model}.npy"))
        print('Done!')

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    i = 0
    lls = labels
    target = "class"

    u_lls = np.unique(lls)
    cmap = plt.cm.tab10

    print('Plotting classes ... ', end="")

    handles = []
    markers = ['8', '*']

    for j, l in enumerate(u_lls):
        fidx = (lls == l)
        color_feat = cmap(j % 10)
        color_w = cmap((j + len(u_lls)) % 10)
        marker_w = markers[j % len(markers)]
        size_w = max(300-j*100,50)

        # Features scatter
        axs[i].scatter(features_2d[fidx, 0], features_2d[fidx, 1], alpha=0.4, s=4, marker="o", color=color_feat, zorder=1)

        # Weights scatter
        axs[i].scatter(weights_2d[j, 0], weights_2d[j, 1], alpha=1.0, s=size_w, marker=marker_w, color=color_w, zorder=2)

        # Create proxy handles for legend
        feature_proxy = mlines.Line2D([], [], color=color_feat, marker="o", linestyle="None",
                                      markersize=6, label=f"feature {target}: {l}")
        weight_proxy = mlines.Line2D([], [], color=color_w, marker=marker_w, linestyle="None",
                                     markersize=14, label=f"weight {target}: {l}")

        handles.append(feature_proxy)
        handles.append(weight_proxy)

    print('Done!')

    axs[i].legend(handles=handles, loc='lower right', ncol=len(u_lls))
    axs[i].set_title(f"{args.method} colored by {target}")

    i += 1
    lls = domains
    target = "domain"

    u_lls = np.unique(lls)
    cmap = plt.cm.tab10

    print('Plotting domains ... ', end="")
    
    handles = []

    for j, l in enumerate(u_lls):
        fidx = (lls == l)
        color_feat = cmap(j % 10)
        # Features scatter
        axs[i].scatter(features_2d[fidx, 0], features_2d[fidx, 1], alpha=0.4, s=4, marker="o", color=color_feat, zorder=1)
        # Create proxy handles for legend
        feature_proxy = mlines.Line2D([], [], color=color_feat, marker="o", linestyle="None",
                                      markersize=6, label=f"feature {target}: {l}")
        handles.append(feature_proxy)

    print('Done')

    axs[i].legend(handles=handles, loc='upper center', ncol=len(u_lls))
    axs[i].set_title(f"{args.method} colored by {target}")
    
    fig.suptitle(f"model: {args.model}")

    plt.savefig(os.path.join(dir, f"{args.method}_{args.model}.jpg"), format='jpg')
    os.startfile(os.path.abspath(os.path.join(dir, f"{args.method}_{args.model}.jpg")))
    
if __name__ == "__main__":
    # create the top-level parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./results/CMNIST/')
    parser.add_argument('--skip_embedding', action='store_true', help='skip embedding; load instead')
    parser.add_argument('--model', type=str, default="val", choices=["val", "best", "last"])
    subparsers = parser.add_subparsers(dest='method', help='embedding method: tsne, pca, umap')

    # create the parser for the "tsne" command
    parser_tsne = subparsers.add_parser('tsne', help='tsne')
    parser_tsne.add_argument('--max_iter', type=int, default=1000, help='at least 250')
    parser_tsne.add_argument('--perplexity', type=int, default=30)

    # create the parser for the "pca" command
    parser_pca = subparsers.add_parser('pca', help='pca')

    # create the parser for the "umap" command
    parser_umap = subparsers.add_parser('umap', help='umap')
    parser_umap.add_argument('--n_neighbors', type=int, default=15)
    parser_umap.add_argument('--min_dist', type=float, default=0.1, help="[0.0,1.0]")
       
    args = parser.parse_args()

    main(args)
