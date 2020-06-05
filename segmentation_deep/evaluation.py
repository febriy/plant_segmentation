import numpy as np

"""calculates dice scores when Scores class for it"""


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / (pred + targs).sum()


""" initialize a empty list when Scores is called, append the list with dice scores
for every batch, at the end of epoch calculates mean of the dice scores"""


class Scores:
    def __init__(self, phase, epoch):
        self.base_dice_scores = []

    def update(self, targets, outputs):
        probs = outputs
        dice = dice_score(probs, targets)
        self.base_dice_scores.append(dice)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        return dice


"""return dice score for epoch when called"""


def epoch_log(epoch_loss, measure):
    """logging the metrics at the end of an epoch"""
    dices = measure.get_metrics()
    dice = dices
    print("Loss: %0.4f |dice: %0.4f" % (epoch_loss, dice))
    return dice
