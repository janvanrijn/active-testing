import matplotlib.pyplot as plt
import numpy as np
import os


def plot_loss_curves(taskid_losscurve, plot_directory, filename):
    try:
        os.makedirs(plot_directory)
    except FileExistsError:
        pass

    max_size = 0

    for task_id, losscurve in taskid_losscurve.items():
        if len(losscurve) > max_size:
            max_size = len(losscurve)
        x = np.arange(len(losscurve))
        plt.step(x, losscurve, where='pre', label='Task %d' %task_id)

    plt.xlim(0, max_size)
    plt.ylim(0, 1)
    plt.savefig(plot_directory + '/' + filename)