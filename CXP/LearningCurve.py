import torch
import matplotlib.pyplot as plt

def PlotLearnignCurve():
    LrCurv_param = torch.load('results/Saved_items')
    batch_size = LrCurv_param['batch_size']
    epoch_losses_train =LrCurv_param['epoch_losses_train']
    epoch_losses_val = LrCurv_param['epoch_losses_val']
    print("batch_size", batch_size)
    #print('best_epoch:', best_epoch)



    plt.figure()
    plt.plot(epoch_losses_train, label="Training Loss")
    plt.plot(epoch_losses_val, label="Validation Loss")
    plt.title("Graph of Epoch Loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()
    #   plt.show() Instead of showing the graph, lets save it it as a png file
    plt.savefig('results/epoch_losses.png')
