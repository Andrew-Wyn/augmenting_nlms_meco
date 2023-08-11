import torch

def mask_mse_loss(b_output, b_target, target_pad, d_out):
    """
    Masks the pad tokens of by setting the corresponding output and target tokens equal.
    """
    active_mask = b_target.view(-1, d_out) == target_pad
    active_outputs = b_output.view(-1, d_out)
    active_targets = torch.where(active_mask, active_outputs, b_target.view(-1, d_out))

    return active_outputs, active_targets


class GazePredictionLoss:
    """
    Loss that deals with a list of variable length sequences. The object call returns global + per-feature MAE loss.
    """

    def __init__(self, d_gaze):
        self.d_gaze = d_gaze
        self.d_report = d_gaze + 1

        self.loss = torch.nn.L1Loss(reduction="sum")

    def __call__(self, b_output, b_target):
        b_length = [len(i) for i in b_output]
        losses = torch.zeros(self.d_report)

        losses[0] = sum([self.loss(i, j) for i, j in zip(b_output, b_target)])
        for output_orig_len, target_orig_len in zip(b_output, b_target):
            for i in range(1, self.d_report):
                losses[i] += self.loss(output_orig_len[:, i - 1], target_orig_len[:, i - 1]).cpu()

        losses[0] /= sum([i * self.d_gaze for i in b_length])
        losses[1:] /= sum(b_length)
        return losses