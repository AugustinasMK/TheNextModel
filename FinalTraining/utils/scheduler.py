import math


def cosine_lr(epoch):
    if epoch < 5:
        return 0.99 * epoch / 5 + 0.01
    elif 5 <= epoch < 10:
        return 1
    else:
        return 0.5 * (math.cos((epoch - 10) / (25 - 10) * math.pi) + 1)
