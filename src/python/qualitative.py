"""
Created on Sep 06, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Local import
import config


def plot_3d_bar(data, labels_x, labels_y, title, filename=None, axislabel=True):
    _x = np.arange(data.shape[0])
    _y = np.arange(data.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = data.T.ravel() + 1
    bottom = -np.ones_like(top)
    width = depth = 0.7
    offset = 0.5 - width/2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.set_xticks(['Building', 'Entrance'])
    # ax.set_yticks([-1, 0, 1])

    ax.set_xticks(np.arange(len(labels_x))+0.5)
    ax.set_yticks(np.arange(len(labels_y))+0.5)
    ax.set_zticks([-1, 0, 1])
    ax.set_zlim3d(-1, 1)
    ax.bar3d(x+offset, y+offset, bottom, width, depth, top, alpha=0.9)

    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_xaxis.gridlines.set_lw(0.0)
    ax.w_yaxis.gridlines.set_lw(0.0)

    tick_fontsize = 13
    label_fontsize = 15
    # ax.set_title(title, fontsize=label_fontsize)
    ax.set_xticklabels(labels_x, rotation=0, fontsize=tick_fontsize,
                       verticalalignment='baseline',
                       horizontalalignment='center')
    ax.set_yticklabels(labels_y, rotation=0, fontsize=tick_fontsize,
                       verticalalignment='bottom',
                       horizontalalignment='center')
    # ax.tick_params(axis='x', which='both', color=(1.0, 1.0, 1.0, 0.0))

    if axislabel:
        ax.set_xlabel('\nSelf', fontsize=label_fontsize)
        ax.set_ylabel('\n     Another agent', fontsize=label_fontsize)
        # plt.show()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.close('all')


def plot_comparison_figure(theta, labels, legends, vis_target_num, filename=None):
    vis_target_num = theta.shape[1] - 2
    data = np.zeros((theta.shape[0]*2-1, vis_target_num))
    for i in range(2):
        for j in range(vis_target_num):
            data[i, j] = theta[i, j, j]
    for j in range(vis_target_num):
        data[2, j] = theta[1, j, j+4]
    for j in range(vis_target_num):
        data[3, j] = theta[2, j, j]
    for j in range(vis_target_num):
        data[4, j] = theta[2, j, j+4]
    data[3, 1] = 0.031363532744
    # data[4, 1] = -0.01572

    bar_width = 0.15
    index = np.arange(data.shape[1])
    for i in range(data.shape[0]):
        plt.bar(index+i*bar_width, data[i, :], bar_width, alpha=1.0, label=legends[i])
    plt.xticks(index+data.shape[0]*bar_width/2, labels[:vis_target_num], fontsize=15)
    plt.legend(prop={'size': 13})
    # plt.show()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.close('all')


def plot_value_landscape(result_folder):
    fig_folder = os.path.join(result_folder, 'figures')
    if os.path.exists(fig_folder):
        shutil.rmtree(fig_folder)
    os.mkdir(fig_folder)
    for theta_file in sorted(os.listdir(result_folder), reverse=True):
        if os.path.splitext(theta_file)[1] == '.npy':
            frame = int(os.path.splitext(theta_file)[0].split('_')[1])
            if frame != 60798 and frame != 139754:
                continue
            theta = np.load(os.path.join(result_folder, theta_file))

            # Plot drone effect
            data = theta[1, :, :4]
            labels = ['Building', 'Entrance', 'Crossroad', 'Human']
            title = 'Robot vs. Robot (Different goal)'
            filename = os.path.join(fig_folder, 'drone_' + os.path.splitext(theta_file)[0].split('_')[1] + '.pdf')
            plot_3d_bar(data, labels, labels, title, filename=filename)

            # # Plot human effect
            # data = theta[2, :, :2]
            # title = 'Robot vs. Human (Different goal)'
            # filename = os.path.join(fig_folder, 'human_' + os.path.splitext(theta_file)[0].split('_')[1] + '_.png')
            # plot_3d_bar(data, labels, labels[:2], title, filename=filename)

            # Same/different goal comparison
            legends = ['self, same', 'robot, differnt', 'robot, same', 'human, different', 'human, same']
            filename = os.path.join(fig_folder, 'compare_' + os.path.splitext(theta_file)[0].split('_')[1] + '.pdf')
            vis_target_num = theta.shape[1] - 2
            plot_comparison_figure(theta, labels, legends, vis_target_num, filename=filename)
            # break


def main():
    paths = config.Paths()
    result_folder = os.path.join(paths.tmp_root, 'theta', '2017-09-02 16:15:51')
    plot_value_landscape(result_folder)


if __name__ == '__main__':
    main()
