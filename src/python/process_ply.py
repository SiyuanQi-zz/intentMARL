"""
Created on Aug 19, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import Queue

import numpy as np
import matplotlib.pyplot as plt
import plyfile
import json

import config


def append_metadata(metadata, ply_path):
    ply_data = plyfile.PlyData.read(ply_path)
    metadata['min'].append([float(np.amin(ply_data['vertex'][dim])) for dim in ['x', 'y', 'z']])
    metadata['max'].append([float(np.amax(ply_data['vertex'][dim])) for dim in ['x', 'y', 'z']])
    metadata['mean'].append([float(np.mean(ply_data['vertex'][dim])) for dim in ['x', 'y', 'z']])
    return


def create_labels(paths):
    labels = dict()
    data_path = os.path.join(paths.project_root, 'mesh')
    for scene in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, scene)):
            labels[scene] = dict()
            objects_in_scene = list()
            for ply in os.listdir(os.path.join(data_path, scene)):
                if os.path.isdir(os.path.join(data_path, scene, ply)):
                    continue

                if os.path.splitext(ply)[1] != '.ply':
                    continue

                obj_name = os.path.splitext(ply)[0].split('_')[0]
                if not obj_name in objects_in_scene:
                    objects_in_scene.append(obj_name)
                    labels[scene][obj_name] = dict()
                    labels[scene][obj_name]['max'] = list()
                    labels[scene][obj_name]['min'] = list()
                    labels[scene][obj_name]['mean'] = list()
                    append_metadata(labels[scene][obj_name], os.path.join(data_path, scene, ply))
                else:
                    append_metadata(labels[scene][obj_name], os.path.join(data_path, scene, ply))

    with open(os.path.join(paths.tmp_root, 'labels.json'), 'w') as f:
        json.dump(labels, f, indent=4, separators=(',', ': '))
    return


def get_pos_height(ply_data, scale, mins):
    # truncated_ply_data = truncate_ply(ply_data)
    positions = np.vstack((((ply_data['vertex'][dim]-mins[dim])*scale).astype(int) for dim in ['x', 'z'])).T
    heights = ply_data['vertex']['y']
    return positions, heights


def cal_height_map(ply_data, labels, scene, scale=1.0):
    height_map = np.zeros((int((labels[scene]['scene']['max'][0][0]-labels[scene]['scene']['min'][0][0])*scale)+1,
                           int((labels[scene]['scene']['max'][0][2]-labels[scene]['scene']['min'][0][2])*scale)+1))

    mins = {'x':labels[scene]['scene']['min'][0][0], 'z':labels[scene]['scene']['min'][0][2]}

    positions, heights = get_pos_height(ply_data, scale, mins)
    # print positions.shape

    for i in range(positions.shape[0]):
        if heights[i] > height_map[positions[i, 0], positions[i, 1]]:
            height_map[positions[i, 0], positions[i, 1]] = heights[i]

    height_map *= scale
    height_map -= 3.9
    height_map[height_map < 0] = 0

    # print height_map.shape
    # fig = plt.figure()
    # ax = plt.imshow(height_map, extent=[0, 1, 0, 1])
    # fig.colorbar(ax)
    # plt.show()

    return height_map


def process_ply(paths, scene, scale):
    ply_data = plyfile.PlyData.read(os.path.join(paths.project_root, 'mesh', scene, 'scene.ply'))

    with open(os.path.join(paths.tmp_root, 'labels.json'), 'r') as f:
        labels = json.load(f)
    height_map = cal_height_map(ply_data, labels, scene, scale)

    height_map_folder = os.path.join(paths.tmp_root, 'height_maps')
    if not os.path.exists(height_map_folder):
        os.makedirs(height_map_folder)

    np.save(os.path.join(height_map_folder, scene), height_map)


def main():
    paths = config.Paths()
    scene = 'city'
    scale = 0.1

    # create_labels(paths)
    process_ply(paths, scene, scale)  # each unit is 10 centimeters


if __name__ == '__main__':
    main()
