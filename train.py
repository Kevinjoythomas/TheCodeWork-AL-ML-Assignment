
from datasets import load_dataset
from create_promts import generate_text_prompt_with_gpt
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
from PIL import Image
from image_dataset import CustomDataset
from transformers import CLIPTextModel
from accelerate import Accelerator
import os

print("Loading dataset...")
# Load dataset
dataset = load_dataset("mrtoy/mobile-ui-design")["train"]
print(f"Dataset loaded: {len(dataset)} samples")
# Prepare prompts
prompts = [generate_text_prompt_with_gpt(item['objects']) for item in dataset]
print(f"Generated {len(prompts)} prompts")
# Prepare images
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

images = [transform(item['image'].convert("RGB")) for item in dataset]



train_dataset = CustomDataset(images, prompts)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Initialize components for Stable Diffusion
model_id = "CompVis/stable-diffusion-v1-4"
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

# Fine-tuning configurations
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
accelerator = Accelerator()
unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
print("Starting training loop")
# Training loop
unet.train()
epochs = 3
for epoch in range(epochs):  # Number of epochs
    print(f"Epoch {epoch + 1}/{epochs} started")
    for images, texts in train_dataloader:
        images = images.to(accelerator.device)
        texts = text_encoder(texts.to(accelerator.device))[0]

        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.size(0),), device=latents.device).long()

        noisy_latents = latents + noise

        # UNet output and loss computation
        pred_noise = unet(noisy_latents, timesteps, texts).sample
        loss = torch.nn.functional.mse_loss(pred_noise, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}: Loss {loss.item()}")

# Save fine-tuned model
unet.save_pretrained("fine_tuned_unet")
