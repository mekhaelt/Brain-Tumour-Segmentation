import torch
import os
from monai.networks.nets import SwinUNETR
from dataloaders import get_loader
from functools import partial
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
import time
from monai.data import decollate_batch
from utils import AverageMeter
import numpy as np
from utils import save_checkpoint
import matplotlib.pyplot as plt


data_dir = 'C:/Users/mekha/Desktop/Brain-Tumour-Detection/raw_data/training_data1_v2'        
json_list = "C:/Users/mekha/Desktop/Brain-Tumour-Detection/Brain-Tumour-Swin_UNETR/dataset.json"            
roi = (128, 128, 128)
batch_size = 2
sw_batch_size = 4                             
fold = 1
infer_overlap = 0.5
max_epochs = 50
val_every = 10
start_epoch = 0

train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, roi)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    in_channels=4,
    out_channels=3,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
).to(device)

torch.backends.cudnn.benchmark = True
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)
dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
model_inferer = partial(
    sliding_window_inference,
    roi_size = [roi[0], roi[1], roi[2]],
    sw_batch_size = sw_batch_size,
    predictor = model,
    overlap = infer_overlap,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


def train_epoch(model, loader, optimizer, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch_data in enumerate(loader):
        data, target = batch_data['image'].to(device), batch_data['label'].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        print(
            'Epoch {}/{} {}/{}'.format(epoch, max_epochs, idx, len(loader)),
            'loss: {:.4f}'.format(run_loss.avg),
            'time {:.2f}s'.format(time.time() - start_time),
        )

        start_time = time.time()
    return run_loss.avg

def val_epoch(model, loader, epoch, acc_func, model_inferer=None, post_sigmoid=None, post_pred=None,):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data['image'].to(device), batch_data['label'].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for  val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                'Val {}/{} {}/{}'.format(epoch, max_epochs, idx, len(loader)),
                ', dice_tc:',
                dice_tc,
                ', dice_wt:',
                dice_wt,
                ', dice_et:',
                dice_et,
                ', time {:.2f}s'.format(time.time() - start_time),
            )
            start_time = time.time()

        return run_acc.avg

def trainer(model, train_loader, val_loader, optimizer, loss_func, acc_func, scheduler, model_inferer=None, start_epoch=0, post_sigmoid=None, post_pred=None,):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), 'Epoch: ', epoch)
        epoch_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, epoch=epoch, loss_func=loss_func)
        print('Final training  {}/{}'.format(epoch, max_epochs - 1), 'loss: {:.4f}'.format(train_loss), 'time {:.2f}s'.format(time.time() - epoch_time))

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(model, val_loader, epoch=epoch, acc_func=acc_func, model_inferer=model_inferer, post_sigmoid=post_sigmoid, post_pred=post_pred,)
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            print(
                'Final validation stats {}/{}'.format(epoch, max_epochs - 1), ', dice_tc:', dice_tc, ', dice_wt:', dice_wt,
                ', dice_et:', dice_et, ', Dice_Avg:', val_avg_acc, ', time {:.2f}s'.format(time.time() - epoch_time),
            )

            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)

            if val_avg_acc > val_acc_max:
                print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(model, epoch, best_acc=val_acc_max)
        scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )

if __name__ == "__main__":
    (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    ) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )
    print('Training Complete')
    print(f"Best Average Dice Score: {val_acc_max:.4f} ")
    print(f"Final Mean Dice Score: {dices_avg[-1]:.4f}")

    # Tumor Core (TC)
    print(f"Best TC Dice Score: {max(dices_tc):.4f}")
    print(f"Mean TC Dice Score: {np.mean(dices_tc):.4f}")

    # Whole Tumor (WT)
    print(f"Best WT Dice Score: {max(dices_wt):.4f}")
    print(f"Mean WT Dice Score: {np.mean(dices_wt):.4f}")

    # Enhancing Tumor (ET)
    print(f"Best ET Dice Score: {max(dices_et):.4f}")
    print(f"Mean ET Dice Score: {np.mean(dices_et):.4f}")



    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, loss_epochs, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_avg, color="green")
    plt.show()
    plt.figure("train", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Val Mean Dice TC")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_tc, color="blue")
    plt.subplot(1, 3, 2)
    plt.title("Val Mean Dice WT")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_wt, color="brown")
    plt.subplot(1, 3, 3)
    plt.title("Val Mean Dice ET")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_et, color="purple")
    plt.show()

