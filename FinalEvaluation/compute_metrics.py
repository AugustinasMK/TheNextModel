import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from isc.io import read_ground_truth, read_predictions
from isc.metrics import evaluate, Metrics, print_metrics


def plot_pr_curve(
        metrics: Metrics, title: str, pr_curve_filepath: Optional[str] = None
):
    _ = plt.figure(figsize=(12, 9))
    plt.plot(metrics.recalls, metrics.precisions)
    plt.xlabel("Atkūrimas")
    plt.ylabel("Preciziškumas")
    plt.title(title)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid()
    if pr_curve_filepath:
        plt.savefig(pr_curve_filepath, format='eps')


def compute_metrics(preds_filepath, gt_filepath, title, norm: bool = False):
    predictions = read_predictions(preds_filepath)
    gts = read_ground_truth(gt_filepath)

    print(
        f"Track 1 results of {len(predictions)} predictions ({len(gts)} GT matches)"
    )
    metrics = evaluate(gts, predictions)
    print_metrics(metrics)

    save_dir = '/'.join(preds_filepath.split('/')[:-1])
    print('Saving')
    np.save(f'{save_dir}/recalls{"_norm" if norm else ""}.npy', metrics.recalls)
    np.save(f'{save_dir}/precisions{"_norm" if norm else ""}.npy', metrics.precisions)
    print('Saved')
    plot_pr_curve(metrics, title, f"{save_dir}/pr{'_norm' if norm else ''}.eps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default='disc', choices=['disc', 'glv2_q', 'glv2_t'])
    parser.add_argument('-m', '--model_type', type=str, default='disc', choices=['disc', 'glv2_q', 'glv2_t'])
    parser.add_argument('-e', '--epoch', type=str, required=True)
    parser.add_argument('-t', '--title', type=str, default='PA kreivė')

    args = parser.parse_args()

    if args.dataset == 'disc':
        preds_dir = f"./data/disc/{args.model_type}/{args.epoch}/"
        gt = './data/disc/ground_truth.csv'
    elif args.dataset == 'glv2_q':
        preds_dir = f"./data/glv2_q/{args.model_type}/{args.epoch}/"
        gt = './data/glv2_q/ground_truth.csv'
    else:
        preds_dir = f"./data/glv2_t/{args.model_type}/{args.epoch}/"
        gt = './data/glv2_t/ground_truth.csv'

    compute_metrics(f"{preds_dir}matrix_no_norm.csv", gt, args.title, False)
    compute_metrics(f"{preds_dir}matrix_norm.csv", gt, f"{args.title} (Normalizuota)", True)
