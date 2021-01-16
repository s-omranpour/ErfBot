import torch
import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, AdamW

class ErfBot(pl.LightningModule):
    def __init__(self, config, lr, weight_decay, max_epochs):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.save_hyperparameters()
        self.model = GPT2LMHeadModel(config)
        
    def forward(self, inputs):
        res = self.model(**inputs, labels=inputs["input_ids"])
        return res.logits, res.loss

    def step(self, batch, mode='train'):
        outputs, losses = self.forward(batch)
        loss = sum([losses[k] for k in losses])
        self.log(mode+'_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch)
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, mode='val')
    
    def test_step(self, batch, batch_idx):
        outputs, _ = self.forward(batch)
        return outputs

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        opt = AdamW(grouped_parameters, lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=1e-8)
        return [opt], [sch]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)