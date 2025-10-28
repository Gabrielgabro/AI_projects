%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

# use GPU for computation if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load code for NVAE

# make sure that files in local directory can be imported
import sys
sys.path.insert(0, ".")

# try to import nvae module and download it if it fails
import urllib
try:
    import nvae
except ImportError:
    urllib.request.urlretrieve("https://uu-sml.github.io/course-apml-public/lab/nvae.py", "nvae.py")
    import nvae

# load pretrained NVAE for MNIST
model = nvae.load_pretrained_model("mnist", use_gdown=True)

# move model to the GPU if available
model = model.to(device)

# number of convolutional layers
len(model.all_conv_layers)

# number of parameters
nvae.count_parameters(model)

# preprocessing pipeline: add padding and sample binarized version
preprocess = transforms.Compose([
    transforms.Pad(2),
    lambda x: x.bernoulli(),
])

# download MNIST test data
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# define data loader
# if you use a GPU you can increase the batch size
# `pin_memory=True` is helpful when working with GPUs: https://pytorch.org/docs/stable/data.html#memory-pinning
testdata = torch.utils.data.DataLoader(
    testset, batch_size=10, shuffle=True, pin_memory=device.type=='cuda',
)

with torch.no_grad(): # no gradients required
    # batch of 10 random images + labels from the test data set
    test_images, test_labels = next(iter(testdata))

    # move data to the GPU if available
    test_images = test_images.to(device)

    # compute reconstructions
    logits = model(preprocess(test_images))[0]
    test_reconstructions = centercrop(model.decoder_output(logits).mean)

# plot a grid of random pairs of `originals` and `reconstructions`
def plot_reconstructions(originals, reconstructions, labels, nrows=4, ncols=2):
    # indices of displayed samples
    n = originals.shape[0]
    indices = np.random.choice(n, size=nrows*ncols, replace=False)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    for (idx, ax) in zip(indices, axes.flat):
        # extract original, reconstruction, and label
        original = originals[idx]
        reconstruction = reconstructions[idx]
        label = labels[idx]

        # configure subplot
        ax.set_xticks(())
        ax.set_yticks(())
        ax.grid(False)
        ax.set_title(f"Label: {label.item()}", fontweight='bold')

        # plot original and reconstructed image in a grid
        grid = np.ones((32, 62))
        grid[2:30, 2:30] = original.squeeze().cpu().numpy()
        grid[2:30, 32:60] = reconstruction.squeeze().cpu().numpy()
        ax.imshow(grid, vmin=0.0, vmax=1.0, cmap='gray_r')

    return fig

# only analyze 10 out of 1000 batches
# you can increase this if you use a GPU
nbatches = 10

sqerrors = []

with torch.no_grad(): # no gradients required
    for i, (images, _) in enumerate(testdata):
        # only analyze nbatches batches
        if i >= nbatches:
            break

        # show progress
        print("processing batch {:3d} ...".format(i + 1))

        # move data to GPU if available
        images = images.to(device)

        # compute reconstructions
        logits = model(preprocess(images))[0]
        reconstructions = centercrop(model.decoder_output(logits).mean)

        # compute average squared reconstruction error
        sqerror = (images - reconstructions).pow(2).view(-1, 784).sum(dim=1).mean()
        sqerrors.append(sqerror)

sqerr = torch.tensor(sqerrors).mean()
print(f"Average squared reconstruction error: {sqerr}")

# sample a single image with temperature `T`
T = 0.6

with torch.no_grad(): # no gradients required
    # compute decoding distribution
    logits = model.sample(1, T)
    output = model.decoder_output(logits)

    # use non-binarized sample for MNIST, otherwise sample from the decoding distribution
    output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample()

    # reorder the axes as (nsamples, channels, height, width)
    output_img = output_img.permute(0, 2, 3, 1).squeeze().cpu().numpy()

    # crop MNIST images
    if model.dataset == 'mnist':
        output_img = output_img[2:30, 2:30]

# plot sample
plt.xticks([])
plt.yticks([])
if model.dataset == 'mnist':
    # plot MNIST images as grayscale images
    plt.imshow(output_img, vmin=0.0, vmax=1.0, cmap='gray_r')
else:
    plt.imshow(output_img)
plt.show()

# load pretrained NVAE
# possible data sets:
# "mnist", "celeba_64", "celeba_256a", "celeba_256b", "cifar10a", "cifar10b", "ffhq"
model = nvae.load_pretrained_model("celeba_256a", use_gdown=True)

# move model to the GPU if available
model = model.to(device)
