import pytorch_lightning as pl
import wandb
from src.dataloader import bgr2GRBG, grbg2LR4
import torch
import numpy as np
from torchmetrics import PeakSignalNoiseRatio
psnr = PeakSignalNoiseRatio()
from PIL import ImageDraw, Image
import PIL
from os.path import join
font = PIL.ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 20)

class ImagePredictionLogger(pl.Callback):
    def __init__(self, config):
        super().__init__()

        imageName = join(config.train_data_path, str(151).zfill(4)) + '.png'
        img_hr = np.flip(np.array(Image.open(imageName)), 2)
        output_patch = img_hr[200:(200 + config.patch_size), 200:(200 + config.patch_size), :]
        output_patch = ((output_patch - np.min(output_patch)) / (
                    np.max(output_patch) - np.min(output_patch)) - 0.5) / 0.5
        if config.bayer == 'grbg':
            input_patch = bgr2GRBG(output_patch)
        else:
            raise ValueError(f'{config.bayer} bayer scheme is not available!!!')

        input_patch_LR4 = grbg2LR4(input_patch)
        x = np.linspace(-1, 1, config.patch_size)
        y = np.linspace(-1, 1, config.patch_size)
        coords_x, coords_y = np.float32(np.meshgrid(x, y))

        input_coords_x = torch.tensor(coords_x.reshape(-1, 1))
        input_coords_y = torch.tensor(coords_y.reshape(-1, 1))
        self.input_patch_LR4 = torch.tensor(input_patch_LR4)
        self.output_patch = torch.tensor(output_patch)
        self.model_input = torch.stack((input_coords_y, input_coords_x), dim=-1).squeeze(1)
        self.patch_size = config.patch_size

    def on_validation_epoch_end(self, trainer, pl_module):
        model_input = self.model_input.to(device=pl_module.device).unsqueeze(0)
        input_patch_LR4 = self.input_patch_LR4.to(device=pl_module.device).unsqueeze(0)
        model_output = pl_module(model_input, input_patch_LR4)

        merge_pil = Image.new('RGB', (2*self.patch_size, self.patch_size))

        gt_imPIL = Image.fromarray(np.asarray((self.output_patch[ :, :, :].flip(2).reshape(-1, self.patch_size,
                                                                                self.patch_size, 3).squeeze(
            0).detach().cpu() + 1) * 127).astype(np.uint8))
        ImageDraw.Draw(gt_imPIL).text((0, 0), 'GT', font=font, fill=255)
        model_outputPIL = Image.fromarray(np.asarray((model_output.flip(2).reshape(-1, self.patch_size,
                                                                                  self.patch_size, 3).squeeze(
            0).detach().cpu() + 1) * 127).astype(np.uint8))
        ImageDraw.Draw(model_outputPIL).text((0, 0), 'Out', font=font, fill=255)

        merge_pil.paste(gt_imPIL, (0, 0))
        merge_pil.paste(model_outputPIL, (self.patch_size, 0))

        trainer.logger.experiment.log({
            "reference_image": wandb.Image(merge_pil),
            "epoch": trainer.current_epoch
        })