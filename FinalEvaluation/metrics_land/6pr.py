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

    disc_precisions = np.load('land_d_precisions.npy')
    disc_recalls = np.load('land_d_recalls.npy')

    glv2q_precisions = np.load('land_q_precisions.npy')
    glv2q_recalls = np.load('land_q_recalls.npy')

    glv2t_precisions = np.load('land_t_precisions.npy')
    glv2t_recalls = np.load('land_t_recalls.npy')

    # plt.plot(gist_recalls, gist_precisions, 'b', label='GIST')
    plt.plot(multi_recalls, multi_precisions, 'g', label='MultiGrain')
    plt.plot(how_recalls, how_precisions, 'r', label='HOW')
    plt.plot(vf_recalls, vf_precisions, 'c', label='VisionForce')
    # plt.plot(resnet_recalls, resnet_precisions, 'm', label='Resnet')
    # plt.plot(inception_recalls, inception_precisions, 'y', label='Inception')
    plt.plot(disc_recalls, disc_precisions, 'k', label='ViT-D')
    plt.plot(glv2t_recalls, glv2t_precisions, 'lime', label='ViT-LT')
    plt.plot(glv2q_recalls, glv2q_precisions, 'orange', label='ViT-LQ')

    plt.xlabel("Atkūrimas")
    plt.ylabel("Preciziškumas")
    plt.xlim(0, 0.55)
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig('6pr.eps', format='eps')