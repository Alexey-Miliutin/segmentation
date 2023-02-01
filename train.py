import argparse
import os

import tqdm
import numpy as np
import torch

from torch.optim import lr_scheduler
from torch import nn
from torch.optim import Adam

from models.unet import UNet
from datasets.dataset import DataLoader
from losses.losses import DiceLoss, dice_coef, jaccard_coef

from utils.wandb_logging import WandbLogger


def save_model(model: nn.Module, epoch: int, model_path: str, wandb_run_id: int, best_val_loss: float) -> None:
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'wandb_run_id': wandb_run_id,
        'best_val_loss': best_val_loss}, model_path)


def train(opt: dict) -> None:
    # create project folder
    if not os.path.exists(os.path.join(opt.project, opt.name)):
        os.makedirs(os.path.join(opt.project, opt.name))
    # init model
    model = UNet(opt.inp_ch, opt.out_ch)
    # init device
    device = torch.device(opt.device)
    # move model to cuda if available
    model.to(device)
    # init lose function
    criterion = DiceLoss()
    # init DataLoaders
    train_loader = DataLoader(opt, 'train')
    val_loader = DataLoader(opt, 'val')
    # init optimizer
    scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=opt.use_amp)
    optimizer = Adam(model.parameters(), lr=opt.lr)
    # init sheduler
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=0.1, last_epoch=-1)
    # run training
    model_path = os.path.join(opt.project, opt.name, opt.name + '_last.pt')
    # restore model from checkpoint
    if os.path.exists(model_path):
        state = torch.load(model_path)
        epoch = state['epoch']
        wandb_run_id = state['wandb_run_id']
        best_val_loss = state['best_val_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}'.format(epoch))
    else:
        epoch = 1
        wandb_run_id = None
        best_val_loss = np.inf
  
    # init wandb logger
    wandb_logger = WandbLogger(opt, opt.name, wandb_run_id)
    wandb_run_id = wandb_logger.run_id

    for epoch in range(epoch, opt.epochs):
        # set train mode
        model.train()
        # init tqdm
        tq = tqdm.tqdm(total=(len(train_loader) * opt.batch_size))
        tq.set_description('Epoch: {}, lr: {}'.format(epoch, scheduler.get_last_lr()[0]))
        
        train_losses = []
        val_losses = []
        jaccard = []
        dice = []

        try:
            for inputs, targets in train_loader:
                # move to cuda if available
                inputs, targets = inputs.to(device), targets.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # automatic mixed precision
                with torch.cuda.amp.autocast(enabled=opt.use_amp):
                    # forward
                    outputs = model(inputs)
                    # calculate train loss
                    loss = criterion(outputs, targets)
                # backward
                scaler.scale(loss).backward()
                # make a step
                scaler.step(optimizer)
                scaler.update()

                train_losses.append(loss.data.item())
                tq.update(opt.batch_size)
                tq.set_postfix(loss='{:.5f}'.format(np.mean(train_losses)))

            scheduler.step(loss)
            mean_train_loss = np.mean(train_losses)
            tq.close()
            save_model(model, epoch+1, model_path, wandb_run_id, best_val_loss)
            # set eval mode
            model.eval()
            # start validation
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # move to cuda if available
                    inputs, targets = inputs.to(device), targets.to(device)
                    # automatic mixed precision
                    with torch.cuda.amp.autocast(enabled=opt.use_amp):
                        # forward
                        outputs = model(inputs)
                    # threshold outputs to calculate metrics
                    # outputs = (outputs > 0.0).float()
                    # calc loss
                    val_losses.append(criterion(outputs, targets).data.item())
                    # calculate jaccard metric
                    jaccard.append(jaccard_coef(outputs, targets).data.cpu().numpy())
                    # calculate dice metric
                    dice.append(dice_coef(outputs, targets).data.cpu().numpy())
                    # calculate mean metrics & mean validation loss
            mean_val_loss = np.mean(val_losses)
            mean_val_jaccard = np.mean(jaccard)
            mean_val_dice = np.mean(dice)

            print('Train loss: {:.7f}, Val loss: {:.7f}, jaccard: {:.7f}, dice: {:.7f}'.format(mean_train_loss, mean_val_loss, mean_val_jaccard, mean_val_dice))
            # wandb logging
            metrics = {"train/loss": mean_train_loss,"val/loss": mean_val_loss, "metric/dice": mean_val_dice, "metric/jaccard": mean_val_jaccard, "lr/LearningRate" : scheduler.get_last_lr()[0]}
            wandb_logger.log(metrics)

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                save_model(model, epoch, model_path.replace('last.pt', 'best.pt'), wandb_run_id, best_val_loss)

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            
            save_model(model, epoch, model_path, wandb_run_id, best_val_loss)
            # clean cuda cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            wandb_logger.finish_run()
            print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Common
    parser.add_argument('--data', default='./data', help='dataset folder')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--use_amp', type=bool, default=False, help='Enable Automatic Mixed Precision')
    # NN
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--inp_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)
    # DataLoader
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    # Wabdb # TODO: add media logging
    parser.add_argument('--project', default='runs/train', help='Project dir')
    parser.add_argument('--name', default='unet', help='Name of experiment')
    parser.add_argument('--entity', default=None, help = 'W&B entity')
    # parser.add_argument('--resume', action='store_true', default=False, help='Resume training') # TODO: resume training
    opt = parser.parse_args()
    print(opt)

    # run training  
    train(opt)
