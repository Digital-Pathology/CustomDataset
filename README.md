# CustomDataset

## Dataset Organization

TODO

## Labels

TODO

## Filtration

When filtration is applied (and filtration results are cached), the length of the dataset will be dynamically updated. To maintain a 1-1 mapping of index $\rightarrow$ region (and a uniform distribution for indexing with uniformly-distributed random indeces), we apply an $O(n)$ dark region indexing policy when a region doesn't pass through filtration.

For an image $I$ with $N$ regions ($len(I)=N$) and only $n$ regions that pass through filtration where $N>=\hat{N}$, dark regions are those of index $n$ where $n>=\hat{N}$ becaue, when filtration is in effect, $len(I)=N-\hat{N}$, so $I[n]$, when $n$ is within the bounds of the length of the image, cannot index regions $(N-\hat{N}) -N$.

If a region $\hat{n}$ does not pass filtration, then selection begins indexing from $n=N$ and iterates by at most $\hat{N}$ towards 0 until a region passing filtration is found.

## Augmentation

TODO 

CustomDataset uses Albumentations for augmentating the regions. See [Albumentations' documentation](https://albumentations.ai/docs/) for details.
