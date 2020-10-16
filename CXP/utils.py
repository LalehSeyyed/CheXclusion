import torch
import os


def Saved_items(epoch_losses_train, epoch_losses_val, time_elapsed, batch_size):
    """
    Saves checkpoint of torchvision model during training.
    Args:

        epoch_losses_train: training losses over epochs
        epoch_losses_val: validation losses over epochs

    """
    print('saving')
    state2 = {
        'epoch_losses_train': epoch_losses_train,
        'epoch_losses_val': epoch_losses_val,
        'time_elapsed': time_elapsed,
        "batch_size": batch_size
    }
    torch.save(state2, 'results/Saved_items')

def checkpoint(model, best_loss, best_epoch, LR):
    """
    Saves checkpoint of torchvision model during training.
    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'LR': LR
    }
    torch.save(state, 'results/checkpoint')

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, model, optimizer, LR, dest_dir):
    """
    Saves model checkpoint.
    :param epoch: epoch number
    :param stn: model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :dest_dir
    """
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer, "learning rate:": LR }
    filename = 'epoch' + str(epoch) + '.pth.tar'
    filename = os.path.join(dest_dir, filename)
    torch.save(state, filename)