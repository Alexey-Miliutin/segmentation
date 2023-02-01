import wandb
from pathlib import Path

class WandbLogger():
    def __init__(self, opt, name, run_id):
        
        self.wandb_run = wandb.init(
            config=opt,
            resume="allow",
            project='Unet' if opt.project == 'runs/train' else Path(opt.project).stem,
            entity=opt.entity,
            name=name,
            id=run_id,
            allow_val_change=True)

        self.run_id = wandb.run.id
        # define metrics
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("val/loss", summary="min")
        wandb.define_metric("metric/dice", summary="max")
        wandb.define_metric("metric/jaccard", summary="max")
        
        # # log config
        # if not opt.resume:
        #   self.wandb_run.config.opt = vars(opt)

    def log(self, log_dict: dict) -> None:
        if self.wandb_run:
            wandb.log(log_dict)

    def finish_run(self) -> None:
        if self.wandb_run:
            wandb.run.finish()