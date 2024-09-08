from datasets import load_dataset
from dataclasses import dataclass
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import UNet2DModel, DDPMPipeline, DDPMScheduler
import torch, os, math
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
from accelerate import notebook_launcher

@dataclass
class TrainingConfig:
    image_size_height = 32  # 生成的图像分辨率
    image_size_width = 64  # 生成的图像分辨率
    train_batch_size = 32
    eval_batch_size = 16  # 评估时采样的图像数量
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 5
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'mcskin_diffuser'  # 生成的模型名称
    push_to_hub = False  # 是否上传保存的模型到HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # 重新运行笔记本时覆盖旧模型
    seed = 0

config = TrainingConfig()

model = UNet2DModel(
    sample_size=(config.image_size_height, config.image_size_width),  # 目标图像分辨率
    in_channels=4,  # 输入通道数，3 for RGB images
    out_channels=4,  # 输出通道数
    layers_per_block=2,  # 每个UNet块使用的ResNet层数
    block_out_channels=(64, 128, 256, 256, 512),  # 每个UNet块的输出通道数
    down_block_types=(
        "DownBlock2D",  # 常规ResNet下采样块
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # 带有空间自注意力的ResNet下采样块
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # 常规ResNet上采样块
        "AttnUpBlock2D",  # 带有空间自注意力的ResNet上采样块
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    ),
)

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGBA")) for image in examples["image"]]
    return {"images": images}

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGBA', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    image_grid = make_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    return images

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (clean_images.shape[0],), device=clean_images.device).long()
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
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)

if __name__ == "__main__":
    dataset = load_dataset("./dataset", split="train")
    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)
