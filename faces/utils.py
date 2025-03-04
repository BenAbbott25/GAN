import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2

class Image_Dataset(Dataset):
    def __init__(self, root_dir: str, image_res: int = 256, grayscale: bool = False):
        self.root_dir = root_dir
        self.image_res = image_res
        self.grayscale = grayscale
        self.images = [os.path.join(root, file)
                       for root, _, files in os.walk(root_dir)
                       for file in files if file.endswith(('.jpg', '.png'))]
        
        # Debug: Print the number of images found
        print(f"Number of images found: {len(self.images)}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        read_flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(image_path, read_flag)
        image = cv2.resize(image, (self.image_res, self.image_res))

        if self.grayscale:
            image_tensor = (torch.tensor(image, dtype=torch.float32) / 255.0).unsqueeze(0)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0
        
        return image_tensor
    
def plot_confidences(fig2,generator_discriminator_output, reference_discriminator_output):
    # print(f"Step {step} - Generator Loss: {generator_loss.item()}, Discriminator Loss: {discriminator_loss.item()}")
    # figure 2
    fig2.clf()
    ax2 = fig2.add_subplot(1, 1, 1)
    x = np.linspace(0, len(generator_discriminator_output), len(generator_discriminator_output))
    generator_discriminator_output = np.concatenate(sorted(generator_discriminator_output.detach().cpu().numpy()))
    reference_discriminator_output = np.concatenate(sorted(reference_discriminator_output.detach().cpu().numpy()))
    ax2.plot(generator_discriminator_output, label="Generator Discriminator Output", color="red")
    ax2.plot(reference_discriminator_output, label="Reference Discriminator Output", color="blue")

    ax2.fill_between(x, generator_discriminator_output, -1, label="Generator Discriminator Output", color="pink")
    ax2.fill_between(x, reference_discriminator_output, 1, label="Reference Discriminator Output", color="lightblue")
    # average
    average_generator_discriminator_output = np.mean(generator_discriminator_output)
    average_reference_discriminator_output = np.mean(reference_discriminator_output)
    ax2.axhline(average_generator_discriminator_output, color="pink", linestyle="--")
    ax2.axhline(average_reference_discriminator_output, color="lightblue", linestyle="--")
    ax2.legend()
    ax2.set_ylim(0, 1)
    # plt.show()
    # plt.pause(0.01)