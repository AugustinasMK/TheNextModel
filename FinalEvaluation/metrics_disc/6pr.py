import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    gist_precisions = np.load('gist_precisions.npy')
    gist_recalls = np.load('gist_recalls.npy')

    multi_precisions = np.load('multi_precisions.npy')
    multi_recalls = np.load('multi_recalls.npy')

    how_precisions = np.load('how_precisions.npy')
    how_recalls = np.load('how_recalls.npy')

    vf_precisions = np.load('vf_precisions.npy')
    vf_recalls = np.load('vf_recalls.npy')

    resnet_precisions = np.load('resnet_precisions.npy')
    resnet_recalls = np.load('resnet_recalls.npy')

    inception_precisions = np.load('inception_precisions.npy')
    inception_recalls = np.load('inception_recalls.npy')

    plt.plot(gist_recalls, gist_precisions, 'r', label='gist')
    plt.plot(multi_recalls, multi_precisions, 'g', label='multi')
    plt.plot(how_recalls, how_precisions, 'b', label='how')
    plt.plot(vf_recalls, vf_precisions, 'y', label='vf')
    plt.plot(resnet_recalls, resnet_precisions, 'c', label='resnet')
    plt.plot(inception_recalls, inception_precisions, 'm', label='inception')

    plt.xlabel("Atkūrimas")
    plt.ylabel("Preciziškumas")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig('6pr.png')