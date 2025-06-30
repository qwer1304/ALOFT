import argparse
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset
from PIL import Image
import pandas as pd
## Progress bar
from tqdm.auto import tqdm


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

class ColoredMNIST(MultipleEnvironmentMNIST):

    def __init__(self, root):
        ENVIRONMENTS = ['p90', 'p85', 'p80', 'p75', 'm90']
        #                                 (root, environments,                dataset_transform,  input_shape,  num_classes)
        super(ColoredMNIST, self).__init__(root, [0.1, 0.15, 0.2, 0.25, 0.9], self.color_dataset, (3, 28, 28,), 2)

        self.input_shape = (3, 28, 28,)
        self.num_classes = 2
        self.N_WORKERS = 1
        self.environments = ENVIRONMENTS

    def color_dataset(self, images, labels, environment):
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        
        images = torch.stack([images, images, torch.zeros_like(images)], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

def main(args):
    # datasets is a list of per-environment TensorDatasets (x,y)
    save_dir_raw = args.output_dir + 'raw/'
    os.makedirs(save_dir_raw, exist_ok=True)
    save_dir_labels = args.output_dir + 'cmnist_label/'
    os.makedirs(save_dir_labels, exist_ok=True)
    save_dir_kfold = args.output_dir + 'kfold/'
    os.makedirs(save_dir_kfold, exist_ok=True)

    datasets = ColoredMNIST(save_dir_raw)

    for d, dataset in tqdm(enumerate(datasets), leave=False, total=len(datasets)):
        save_dir_domain = save_dir_kfold + datasets.environments[d] + '/' 
        os.makedirs(save_dir_domain, exist_ok=True)
        all_filenames = []
        all_labels = []

        for idx, (img_tensor, label) in tqdm(enumerate(dataset), desc=f"Dataset {datasets.environments[d]}", leave=False, total=len(dataset)):
            label += 1 # Label is expected to be 1..N
            save_dir_domain_label = save_dir_domain + str(label.item()) + '/'
            os.makedirs(save_dir_domain_label, exist_ok=True)

            # Create filename
            filename = f"{idx:05d}.jpg"
            filepath = os.path.join(save_dir_domain_label, filename)
            domain_label_file = os.path.join(f"{datasets.environments[d]}", f"{label.item()}", filename)

            if not args.skip_image_creation:
                if args.target_image_size is not None:
                    resize = transforms.Resize((args.target_image_size, args.target_image_size))
                    img_tensor = resize(img_tensor)

                # Convert to PIL image
                pil_img = transforms.ToPILImage()(img_tensor)

                # Save using PIL
                if False and pil_img.mode == 'LA':
                    pil_img = pil_img.convert('RGB')
                pil_img.save(filepath, "JPEG")

            assert(os.path.exists(filepath), f"File {filepath} does not exist!")
            # Accumulate for CSV
            all_filenames.append(domain_label_file)
            all_labels.append(label.item())

        tr_len = int(len(all_filenames) * 6 / 7) # 60000 - training, 10000 - testing
        train_fp = os.path.join(save_dir_labels, datasets.environments[d]+'_train_kfold.txt')
        val_fp = os.path.join(save_dir_labels, datasets.environments[d]+'_crossval_kfold.txt')
        if args.val_domains_only is None:
            # Create a training dataframe
            df = pd.DataFrame({
                "filename": all_filenames[:tr_len],
                "label":    all_labels[:tr_len],
            })
            with open(train_fp, 'w') as f:
                df.to_csv(f, sep=' ', header=False, index=False)

            # Create a crossval dataframe
            df = pd.DataFrame({
                "filename": all_filenames[tr_len:],
                "label":    all_labels[tr_len:],
            })
            with open(val_fp, 'w') as f:
                df.to_csv(f, sep=' ', header=False, index=False)           
        else:
            if datasets.environments[d] not in args.val_domains_only:
                # Create a training dataframe
                df = pd.DataFrame({
                    "filename": all_filenames,
                    "label":    all_labels,
                })
                with open(train_fp, 'w') as f:
                    df.to_csv(f, sep=' ', header=False, index=False)
                open(val_fp, "w").close()
            else:
                # Create a crossval dataframe
                df = pd.DataFrame({
                    "filename": all_filenames,
                    "label":    all_labels,
                })
                open(train_fp, "w").close()
                with open(val_fp, 'w') as f:
                    df.to_csv(f, sep=' ', header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CMNIST dataset')
    parser.add_argument('--output_dir', type=str, default="./data/DataSets/CMNIST/")
    parser.add_argument('--target_image_size', type=int, default=64)
    parser.add_argument('--skip_image_creation', action='store_true')
    parser.add_argument('--val_domains_only', type=str, nargs='+', default=None, help='Use this to assign some domains ONLY as validation ones.')
    args = parser.parse_args()
    
    main(args)


