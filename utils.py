import torch
import config
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os
def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        # save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        # save_image(y * 0.5 + 0.5, folder + f"/tar_{epoch}.png")
        # save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        save_images(x, y, y_fake, epoch, folder)
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_images(input_img, tar_img, gen_img, epoch, folder):
    display_list = [input_img[0], tar_img[0], gen_img[0]]
    title = ['Input Image', 'Ground Truth', 'Generated']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.

        img_np = (display_list[i].cpu() * 0.5 + 0.5).numpy()
        plt.imshow(np.transpose(img_np, (1, 2, 0)))
        plt.axis('off')
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    filename = f"epoch_{epoch+1}"
    # Save the figure to the specified folder
    plt.savefig(os.path.join(folder, filename))
    plt.clf()