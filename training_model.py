# https://github.com/chunyu-li/ddpm/blob/master/sampling.py

from forward_noising import forward_diffusion_sample
from unet import SimpleUnet
from dataloader import load_transformed_dataset
import torch.nn.functional as F
import torch
from torch.optim import Adam
import logging
import torchvision
from torch.utils.data import DataLoader

from dataloader import pedCls_Dataset, show_tensor_image

logging.basicConfig(level=logging.INFO)


def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    # return F.l1_loss(noise, noise_pred)
    return F.mse_loss(noise, noise_pred)


if __name__ == "__main__":
    model = SimpleUnet()
    T = 300     # 预定义的步数，也就是加多少步噪音
    BATCH_SIZE = 32
    epochs = 5

    train_dataset = pedCls_Dataset(ds_name_list=['D4'], txt_name='augmentation_train.txt', img_size=64, get_num=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        # for batch_idx, images in enumerate(train_loader):
        for batch_idx, (batch, _) in enumerate(train_loader):

            optimizer.zero_grad()
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch, t, device=device)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Batch index {batch_idx:03d} Loss: {loss.item()}")

            # break
        # break

    torch.save(model.state_dict(), "./ddpm_mse_epochs_100_D4.pth")
