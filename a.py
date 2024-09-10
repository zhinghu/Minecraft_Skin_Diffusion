import os
import yaml
import math
import torch
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from accelerate import Accelerator
from dataclasses import dataclass, field
from accelerate import notebook_launcher
from diffusers import UNet2DModel, DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# 读取配置文件
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['learning_rate'] = float(config['learning_rate'])
    return config

# 选择设备
def select_device():
    accelerator = Accelerator()
    return accelerator.device

# 定义模型
class UNet2DModelWrapper:
    def __init__(self, config):
        self.model = UNet2DModel(
            sample_size=(config['image_size_height'], config['image_size_width']),
            in_channels=3,  # Minecraft 皮肤通常是 RGB
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
        ).to(config['device'])

    def __call__(self, noisy_images, timesteps, return_dict=False):
        return self.model(noisy_images, timesteps, return_dict=return_dict)

# 数据预处理
def preprocess():
    return transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图像尺寸为 64x64
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

# 转换函数
def transform(examples):
    images = [preprocess()(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

# 创建图像网格
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

# 评估函数
def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config['eval_batch_size'],
        generator=torch.manual_seed(config['seed']),
    ).images

    image_grid = make_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config['output_dir'], "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    return images

# 训练循环
def train_loop(config, model_wrapper, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        log_with="tensorboard"
    )
    if accelerator.is_main_process:
        os.makedirs(config['output_dir'], exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model_wrapper.model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config['num_epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images'].to(config['device'])
            noise = torch.randn(clean_images.shape).to(config['device'])
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (clean_images.shape[0],), device=config['device']).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

            if (epoch + 1) % config['save_image_epochs'] == 0 or epoch == config['num_epochs'] - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config['save_model_epochs'] == 0 or epoch == config['num_epochs'] - 1:
                pipeline.save_pretrained(config['output_dir'])

# 主函数
if __name__ == "__main__":
    config = load_config()
    device = select_device()
    config['device'] = device

    dataset = load_dataset("./dataset", split="train")
    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(UNet2DModelWrapper(config).model.parameters(), lr=config['learning_rate'])

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config['lr_warmup_steps'],
        num_training_steps=(len(train_dataloader) * config['num_epochs']),
    )

    model_wrapper = UNet2DModelWrapper(config)
    args = (config, model_wrapper, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)
