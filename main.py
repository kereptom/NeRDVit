import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.dataloader import PatchDataset
from src.model import SirenLayer, EncoderToModulation
from src.callback import ImagePredictionLogger
from torch import nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchmetrics import PeakSignalNoiseRatio
psnr = PeakSignalNoiseRatio()
from PIL import Image, ImageDraw
import PIL

# ___  ____ ____ ____ _  _ ____ 
# |__] |__| |__/ |__| |\/| [__  
# |    |  | |  \ |  | |  | ___]


hyperparameter_defaults = dict(
    # Train dataset ------------------------------
    train_data_path='datasets/DIV2K/DIV2K_train_HR',
    train_image_ind=[x for x in range(1,801)],
    train_dataset_size=100000,                                 #epoch size
    train_batch_size=13,
    patch_size=100,
    bayer='grbg',
    # Test dataset ------------------------------
    test_data_path='datasets/DIV2K/DIV2K_valid_HR',
    test_image_ind=[804],
    test_dataset_size=5,
    test_batch_size=5,
    # Siren MLP ---------------------------------
    hidden_features=512,
    hidden_layers=10,
    first_omega_0=30,
    hidden_omega_0=30.,
    modulation_scale=100,
    use_bias=True,
    # Encoder-modulator -------------------------
    kernel_size=15,
    encoder_linear_layers=1,
    latent_channels=16,
    # Learning ----------------------------------
    max_epochs=100,
    learning_rate=0.000028649893094086237,
    momentum=0.99,
    optimize="adam0",
    train_loss_cut=2,
    )

wandb.init(config=hyperparameter_defaults,project="NeRDVit")
config = wandb.config
font = PIL.ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 20)

# ____ _   _ ____ ___ ____ _  _ 
# [__   \_/  [__   |  |___ |\/| 
# ___]   |   ___]  |  |___ |  |


class NeRD(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.dim_in = 2
        self.dim_out = 3
        self.dim_hidden = config.hidden_features
        self.num_layers = config.hidden_layers
        self.w0_initial = config.first_omega_0
        self.w0 = config.hidden_omega_0
        self.use_bias = config.use_bias
        self.encoder_linear_layers = config.encoder_linear_layers
        self.kernel_size = config.kernel_size
        self.patch_size = config.patch_size
        self.latent_channels = config.latent_channels
        self.modulation_scale = config.modulation_scale
        self.num_modulations = self.dim_hidden * (self.num_layers - 1)
        self.optimize = config.optimize
        self.lr = config.learning_rate
        self.momentum = config.momentum
        self.train_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.test_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.valid_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.LOSS_MSE = nn.MSELoss()
        self.test_batch_size = config.test_batch_size
        self.train_loss_cut = config.train_loss_cut

        layers = []
        for ind in range(self.num_layers - 1):
            is_first = ind == 0
            layer_w0 = self.w0_initial if is_first else self.w0
            layer_dim_in = self.dim_in if is_first else self.dim_hidden
            layers.append(SirenLayer(dim_in=layer_dim_in, dim_out=self.dim_hidden, w0=layer_w0,
                                     use_bias=self.use_bias, is_first=is_first))

        self.mlp = nn.Sequential(*layers)
        self.last_layer = SirenLayer(dim_in=self.dim_hidden, dim_out=self.dim_out,
                                     w0=self.w0, use_bias=self.use_bias, is_last=True)

        self.encoder = EncoderToModulation(self.num_modulations, self.encoder_linear_layers,
                                                 self.kernel_size, self.patch_size, self.latent_channels)

    def forward(self, x, im_bayer_LR4):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        modulations = self.encoder(im_bayer_LR4)

        idx = 0
        for module in self.mlp:
            shift = modulations[:, idx : idx + self.dim_hidden].unsqueeze(1)
            x = module.linear(x)
            x = x + shift/self.modulation_scale
            x = module.activation(x)
            idx = idx + self.dim_hidden

        out = self.last_layer(x)
        return out.view(*x_shape, out.shape[-1])

    def loss(self, model_input, input_patch_LR4, output_patch):
        model_output = self(model_input, input_patch_LR4)  # this calls self.forward
        model_output = model_output.reshape(-1, config.patch_size, config.patch_size, 3)
        loss = ((model_output - output_patch)**2).mean()
        return loss, model_output


    def configure_optimizers(self):
        if self.optimize == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        if self.optimize == "sgd0":
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        if self.optimize == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.momentum, 0.999))
        if self.optimize == "adam0":
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        model_input, _, input_patch_LR4, output_patch = batch[0], batch[1], batch[2], batch[3]
        loss, outputs = self.loss(model_input, input_patch_LR4, output_patch)
        assert not np.any(np.isnan(loss.detach().cpu().numpy())), f"Loss contained NaN: {loss}"
        assert loss.detach().cpu().numpy() < self.train_loss_cut, f"Loss huge: {loss}"
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return {'loss': loss, 'outputs': outputs, 'targets': output_patch}

    def training_step_end(self, outputs):
        self.train_psnr(outputs['outputs'], outputs['targets'])
        self.log('train_psnr', self.train_psnr, on_epoch=True, on_step=False)

    def training_epoch_end(self, outputs):
        merge_pil = Image.new('RGB', (2*self.patch_size, self.patch_size))

        gt_imPIL = Image.fromarray(np.asarray((outputs[-1]['targets'][1, :, :, :].flip(2).reshape(-1, self.patch_size,
                                                                                self.patch_size, 3).squeeze(
            0).detach().cpu() + 1) * 127).astype(np.uint8))
        ImageDraw.Draw(gt_imPIL).text((0, 0), 'GT', font=font, fill=255)
        model_outputPIL = Image.fromarray(np.asarray((outputs[-1]['outputs'][1, :, :, :].flip(2).reshape(-1, self.patch_size,
                                                                                  self.patch_size, 3).squeeze(
            0).detach().cpu() + 1) * 127).astype(np.uint8))
        ImageDraw.Draw(model_outputPIL).text((0, 0), 'Out', font=font, fill=255)

        merge_pil.paste(gt_imPIL, (0, 0))
        merge_pil.paste(model_outputPIL, (self.patch_size, 0))

        trainer.logger.experiment.log({
            "train_image": wandb.Image(merge_pil),
            "epoch": trainer.current_epoch
        })

    def validation_step(self, batch, batch_idx):
        model_input, _, input_patch_LR4, output_patch = batch[0], batch[1], batch[2], batch[3]
        loss, outputs = self.loss(model_input, input_patch_LR4, output_patch)
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        return {'loss': loss, 'outputs': outputs, 'targets': output_patch}

    def validation_step_end(self, outputs):
        self.valid_psnr(outputs['outputs'], outputs['targets'])
        self.log('valid_psnr', self.valid_psnr, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        model_input, _, input_patch_LR4, output_patch = batch[0], batch[1], batch[2], batch[3]
        loss, outputs = self.loss(model_input, input_patch_LR4, output_patch)
        self.test_psnr(outputs, output_patch)
        self.log('test_psnr', self.test_psnr)
        self.log('test_loss', loss, on_epoch=True)

        merge_pil = Image.new('RGB', (self.patch_size * self.test_batch_size, self.patch_size*2))

        for i in range(self.test_batch_size):

            gt_imPIL = Image.fromarray(np.asarray((output_patch[i,:,:,:].flip(2).reshape(-1, self.patch_size, self.patch_size, 3).squeeze(0).detach().cpu()+1)* 127).astype(np.uint8))
            ImageDraw.Draw(gt_imPIL).text((0, 0), 'GT', font=font, fill=255)
            model_outputPIL = Image.fromarray(np.asarray((outputs[i,:,:,:].flip(2).reshape(-1, self.patch_size, self.patch_size, 3).squeeze(0).detach().cpu()+1)* 127).astype(np.uint8))
            ImageDraw.Draw(model_outputPIL).text((0, 0), 'Out', font=font, fill=255)

            merge_pil.paste(gt_imPIL, (i*self.patch_size, 0))
            merge_pil.paste(model_outputPIL, (i*self.patch_size, self.patch_size))

        trainer.logger.experiment.log({
            "test_image": wandb.Image(merge_pil),
            "epoch": trainer.current_epoch
        })


# ___  ____ ___ ____
# |  \ |__|  |  |__| 
# |__/ |  |  |  |  |

class NeRDDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        self.train_data_path = config.train_data_path
        self.train_image_ind = config.train_image_ind
        self.test_image_ind = config.test_image_ind
        self.test_data_path = config.test_data_path
        self.patch_size = config.patch_size
        self.train_dataset_size = config.train_dataset_size
        self.test_dataset_size = config.test_dataset_size
        self.bayer = config.bayer

    def setup(self, stage=None):
        self.data_train = PatchDataset(self.train_data_path, self.train_image_ind, self.patch_size,
                                       self.train_dataset_size, config.bayer)
        self.data_test = PatchDataset(self.test_data_path, self.test_image_ind, self.patch_size,
                                       self.test_dataset_size, self.bayer)

    def train_dataloader(self):
        train_loader = DataLoader(self.data_train, batch_size=self.train_batch_size, num_workers=32)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.data_test, batch_size=self.test_batch_size, num_workers=32)
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.data_test, batch_size=self.test_batch_size)
        return test_loader

# ___ ____ ____ _ _  _ 
#  |  |__/ |__| | |\ | 
#  |  |  \ |  | | | \|

print(f'\nStarting- a run with hyperparameters:')
for key, value in config.items():
    print('\t', key, ' : ', value)

# setup data
NeRD_data = NeRDDataModule(config)
NeRD_data.setup()

# setup model
NeRD_model = NeRD(config)

# setup wandb
wandb_logger = WandbLogger()
wandb_logger.watch(NeRD_model, log_graph=False)

# fit the model
trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", devices=1, log_every_n_steps=config.train_dataset_size//config.train_batch_size,
                     auto_select_gpus=False, max_epochs=config.max_epochs,callbacks=[ImagePredictionLogger(config)])
trainer.fit(NeRD_model, NeRD_data)

trainer.test(datamodule=NeRD_data)

