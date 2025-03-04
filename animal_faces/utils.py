import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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
            image_tensor = (((torch.tensor(image, dtype=torch.float32) / 255.0) - 0.5) * 2).unsqueeze(0)
        else:
            image_tensor = (((torch.tensor(image, dtype=torch.float32) / 255.0) - 0.5) * 2).permute(1, 0, 2)
        
        return image_tensor

def view_image(image: torch.Tensor, grayscale: bool = False):
    if not hasattr(view_image, "fig"):
        view_image.fig, view_image.axes = plt.subplots(1, 1)

    if grayscale:
        image = image[0].detach().squeeze(0).numpy()
        view_image.axes.imshow(normalise(image), cmap='gray')
    else:
        image = image[0].detach().permute(1, 2, 0).numpy()
        view_image.axes.imshow(normalise(image))
    
    view_image.axes.axis('off')
    view_image.fig.canvas.draw()
    plt.show()
    plt.pause(2)

# def save_image(image: torch.Tensor, filename: str, grayscale: bool = False):
#     if grayscale:
#         image = image[0].detach().squeeze(0).numpy()
#     else:
#         image = image[0].detach().permute(1, 2, 0).numpy()
#     cv2.imwrite(filename, image)

def tensor_to_csv(image: torch.Tensor, filename: str):
    image = image[0].detach().permute(1, 2, 0).numpy()
    np.savetxt(filename, image, delimiter=',')

def normalise(image: np.ndarray) -> np.ndarray:
    image_max = np.max(image)
    image_min = np.min(image)
    image = (image - image_min) / (image_max - image_min)
    random_value = np.random.rand()
    if random_value < 0.2:
        print("sigmoid")
        return 1 / (1 + np.exp(-image))
    else:
        print("clipping")
        return np.clip(np.floor(image * 255).astype(np.int32), 0, 255)
    

def view_image_normalised(image: torch.Tensor):
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for channel in range(image.shape[2]):
                image[row, col, channel] = (image[row, col, channel] + 1) / 2
        
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.pause(0.1)