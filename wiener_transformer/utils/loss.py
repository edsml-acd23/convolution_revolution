import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import altair as alt


import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """
    Automatically weighted multi-task loss.

    Params:
        num: int
            The number of loss functions to combine.
        x: tuple
            A tuple containing multiple task losses.

    Examples:
        loss1 = 1
        loss2 = 2
        awl = AutomaticWeightedLoss(2)
        loss_sum, weighted_losses = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # Initialize parameters for weighting each loss, with gradients enabled
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *losses):
        """
        Forward pass to compute the combined loss.

        Args:
            *losses: Variable length argument list of individual loss values.

        Returns:
            torch.Tensor: The combined weighted loss.
            list[torch.Tensor]: The list of weighted individual loss components.
        """
        loss_sum = 0
        weighted_losses = []

        for i, loss in enumerate(losses):
            # Compute the weighted loss component for each task
            weighted_loss = 0.5 / (self.params[i] ** 2) * loss
            # Add a regularization term to encourage the learning of useful weights
            regularization = torch.log(1 + self.params[i] ** 2)
            # Sum the weighted loss and the regularization term
            weighted_loss_with_reg = weighted_loss + regularization
            loss_sum += weighted_loss_with_reg
            # Store the weighted loss component
            weighted_losses.append(weighted_loss_with_reg)

        return loss_sum, weighted_losses


class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data
