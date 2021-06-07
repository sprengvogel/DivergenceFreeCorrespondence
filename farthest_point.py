import numpy
import cupy


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (num_point, coord_dim)
        y (numpy.ndarray): (num_point, coord_dim)
    Returns (numpy.ndarray): (num_point,)
    """
    return ((x - y) ** 2).sum(axis=1)

def euclid_norm(x,y):
    return numpy.linalg.norm(x-y, axis=-1)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=euclid_norm,
                            skip_initial=False, indices_dtype=numpy.int32,
                            distances_dtype=numpy.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 1-dim array (k, )
            indices of sampled farthest points.
        distances (numpy.ndarray or cupy.ndarray): 2-dim array
            (k, num_point)
    """
    assert pts.ndim == 2
    #print(type(pts))
    xp = cupy.get_array_module(pts)
    num_point, coord_dim = pts.shape
    indices = xp.zeros(k, dtype=indices_dtype)

    # distances[i, j] is distance between i-th farthest point `pts[i]`
    # and j-th input point `pts[j]`.
    distances = xp.zeros((k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[0] = xp.random.randint(len(pts))
    else:
        indices[0] = initial_idx

    farthest_point = pts[indices[0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[0] = xp.argmax(min_distances)
        farthest_point = pts[indices[0]]
        min_distances = metrics(farthest_point[None, :], pts)

    #print(type(distances))
    #print(type(min_distances))
    distances[0, :] = min_distances
    for i in range(1, k):
        indices[i] = xp.argmax(min_distances)
        farthest_point = pts[indices[i]]
        dist = metrics(farthest_point[None, :], pts)
        distances[i, :] = dist
        min_distances = xp.minimum(min_distances, dist)
    return indices


if __name__ == '__main__':
    # when num_point = 10000 & k = 1000 & batch_size = 32,
    # CPU takes 6 sec, GPU takes 0.5 sec.

    from contextlib import contextmanager
    from time import time

    @contextmanager
    def timer(name):
        t0 = time()
        yield
        t1 = time()
        print('[{}] done in {:.3f} s'.format(name, t1-t0))

    num_point = 300000
    coord_dim = 3
    k = 3000
    do_plot = True

    device = 1
    print('num_point', num_point, 'device', device)
    if device == -1:
        pts = numpy.random.uniform(0, 1, (num_point, coord_dim))
    else:
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        pts = cupy.random.uniform(0, 1, (num_point, coord_dim))

    with timer('1st'):
        farthest_indices = farthest_point_sampling(pts, k)

    with timer('2nd'):  # time measuring twice.
        farthest_indices = farthest_point_sampling(pts, k)

    with timer('3rd'):  # time measuring twice.
        farthest_indices = farthest_point_sampling(
            pts, k, skip_initial=True)

    # with timer('gpu'):
    #     farthest_indices = farthest_point_sampling_gpu(pts, k)
    print('farthest_indices', farthest_indices.shape, type(farthest_indices))

    if do_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os
        pts = cupy.asnumpy(pts)
        farthest_indices = cupy.asnumpy(farthest_indices)
        if not os.path.exists('results'):
            os.mkdir('results')
        fig, ax = plt.subplots()
        plt.grid(False)
        plt.scatter(pts[:, 0], pts[:, 1], c='k', s=4)
        plt.scatter(pts[farthest_indices[0]], pts[farthest_indices[1]], c='r', s=4)
        # plt.show()
        plt.savefig('results/farthest_point_sampling.png')