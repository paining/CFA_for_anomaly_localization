import torch
from typing import Union, Tuple, List
import numpy as np
from tqdm import tqdm


def coreset_subsampling_gpu(
    embedding_array: torch.Tensor,
    sampling_number: int,
    gpu_id: int = 0,
    min_distance: Union[float, None] = None,
) -> Tuple[List[int], float]:
    """Coreset Subsamping.
    ---------------------------

    Input Parameter:
    ---------------------------
    embedding_array : torch.Tensor :
        N(number of feature) x D(dimension of feature)

    sampling_number : int :
        number of coreset,

    Return:
    ---------------------------
    cluster_centers : List[int] :
        indexes of sampling cluster centroid.
    max_distance : int :
        maximum distance between cluster centroid.
    """
    if min_distance is None:
        min_distance = 0

    if gpu_id == None:
        device = embedding_array.device
    elif isinstance(gpu_id, int) and gpu_id < torch.cuda.device_count():
        device = torch.device(gpu_id)
    else:
        print("Cannot select proper device. Use CPU")
        device = torch.device("cpu")

    # Coreset Subsampling
    embedding_array = embedding_array.to(device)
    N = embedding_array.shape[0]
    max_distance = 1e10
    min_distances = torch.Tensor(size=(N, 0)).to(device)

    cluster_centers = []
    np.random.seed(0)
    ind = np.random.choice(np.arange(N))
    with tqdm(
        range(sampling_number),
        ncols=79,
        desc="|coreset| Sampling...",
        leave=False,
        dynamic_ncols=True,
    ) as t:
        for i in t:
            cluster_centers.append(ind)
            # x.shape = (i, D)
            x = embedding_array[ind, :]
            # calculate distance from samples to center of clusters.
            # dist.shape = (N, i)
            dist = torch.cdist(
                embedding_array.unsqueeze(0), x.unsqueeze(0)
            ).squeeze(0)

            # calculate minimum distance from center of clusters.
            min_distances = torch.min(
                torch.concat([min_distances, dist], dim=1), dim=1, keepdim=True
            ).values

            # find maximum distance sample.
            ind = torch.argmax(min_distances).item()
            # ind = cupy.argmax(self.min_distances)
            max_distance = min_distances[ind].item()

            if max_distance < min_distance:
                break
        t.set_description(f"|coreset| reach to min_distance({min_distance})")

    return (cluster_centers, max_distance)
