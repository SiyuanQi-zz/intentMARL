"""
Created on Aug 20, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import Queue

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py

import config


def get_best_neighbor(cost_map, pos):
    xlim, ylim = cost_map.shape
    best_pos = pos
    min_cost = np.inf
    for x_step in [-1, 0, 1]:
        for y_step in [-1, 0, 1]:
            if not (0 <= pos[0] < xlim and 0 <= pos[1] < ylim):
                continue
            next_pos = [pos[0] + x_step, pos[1] + y_step]
            if cost_map[next_pos[0], next_pos[1]] < min_cost:
                min_cost = cost_map[next_pos[0], next_pos[1]]
                best_pos = next_pos
    return best_pos


def path_backtrace(cost_map, end_pos):
    if np.isinf(cost_map[end_pos[0], end_pos[1]]):
        return

    path = list()
    path.append(end_pos)
    current_pos = end_pos[:]
    while cost_map[current_pos[0], current_pos[1]] > 0:
        current_pos = get_best_neighbor(cost_map, current_pos)
        path.append(current_pos)
    return path


def plot_path(height_map, path):
    thresh = 0.02
    dilate_size = 2
    blur_sigma = 4

    height_map = height_map.T  # Transpose the image.
    height_map_color = matplotlib.cm.Blues(height_map)

    heat_map = np.zeros_like(height_map)
    for pos in path:
        heat_map[pos[1], pos[0]] = 1

    heat_map = scipy.ndimage.grey_dilation(heat_map, size=(dilate_size, dilate_size))
    heat_map = scipy.ndimage.gaussian_filter(heat_map, sigma=(blur_sigma, blur_sigma))
    heat_map[height_map > 0.5] = 0
    heat_map_color = matplotlib.cm.Reds(heat_map)
    height_map_color[heat_map > thresh, :] = heat_map_color[heat_map > thresh, :]

    plt.imshow(height_map_color, origin='lower')
    plt.show()
    plt.close()

    return height_map_color


def flood_fill(walk_map, start_pos, start_cost=0):
    xlim, ylim = walk_map.shape
    cost_map = np.ones(walk_map.shape) * np.inf

    q = Queue.Queue()
    q.put((start_pos, start_cost))
    while not q.empty():
        pos, cost = q.get()
        if not (0 <= pos[0] < xlim and 0 <= pos[1] < ylim):
            continue
        if walk_map[pos[0], pos[1]] and cost < cost_map[pos[0], pos[1]]:
            cost_map[pos[0], pos[1]] = cost
            for x_step in [-1, 0, 1]:
                for y_step in [-1, 0, 1]:
                    q.put(([pos[0]+x_step, pos[1]+y_step], cost+np.sqrt(x_step**2+y_step**2)))
                    # q.put(([pos[0]+x_step, pos[1]+y_step], cost+np.sqrt(1)))  # Faster, for testing

    return cost_map


def get_walk_map(paths, scene, height_threshold):
    height_map = np.load(os.path.join(paths.tmp_root, 'height_maps', scene+'.npy'))
    height_map = np.flip(height_map.T, 0).T  # Convert coordinates in V-REP to numpy
    walk_map = np.ones(height_map.shape)
    walk_map[height_map >= height_threshold] = 0.0
    return height_map, walk_map


def compute_all_cost_maps(paths, scene, height_threshold=0):
    height_map, walk_map = get_walk_map(paths, scene, height_threshold)
    all_cost_maps = np.empty((height_map.shape[0], height_map.shape[1], height_map.shape[0], height_map.shape[1]))

    for x in range(walk_map.shape[0]):
        for y in range(walk_map.shape[1]):
            print(x, y)
            if walk_map[x, y]:
                cost_map = flood_fill(walk_map, [x, y])
                all_cost_maps[x, y, :, :] = cost_map
                # fig = plt.figure()
                # ax = plt.imshow(cost_map)
                # fig.colorbar(ax)
                # plt.show(ax)
                # plt.close()
            # break
        # break

    cost_map_folder = os.path.join(paths.tmp_root, 'cost_maps')
    if not os.path.exists(cost_map_folder):
        os.makedirs(cost_map_folder)

    h5file = h5py.File(os.path.join(cost_map_folder, scene+'_all.h5'), 'w')
    h5file.create_dataset('all_cost_maps', data=all_cost_maps)
    h5file.close()

    np.save(os.path.join(cost_map_folder, scene+'_all'), all_cost_maps)


def path_planning_demo(paths, scene, height_threshold=0):
    start_pos = [10, 10]
    dest_pos = [141, 110]

    # Read from .h5 file
    # h5file = h5py.File(os.path.join(paths.tmp_root, 'cost_maps', scene+'_all.h5'), 'r')
    # all_cost_maps = h5file['all_cost_maps'][:]
    # h5file.close()

    # Read from .npy file
    all_cost_maps = np.load(os.path.join(paths.tmp_root, 'cost_maps', scene+'_all.npy'))
    height_map, walk_map = get_walk_map(paths, scene, height_threshold)
    path = path_backtrace(all_cost_maps[start_pos[0], start_pos[1], :, :], dest_pos)
    map_with_paths = plot_path(height_map, path)


def main():
    paths = config.Paths()
    scene = 'city'
    height_threshold = 1

    start_time = time.time()
    # compute_all_cost_maps(paths, scene, height_threshold)
    # print('Cost maps computed. Time elapsed: {}s'.format(time.time() - start_time))
    path_planning_demo(paths, scene, height_threshold)
    print('Time elapsed: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
