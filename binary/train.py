from generator import Generator
from discriminator import Discriminator
import math
import torch
import numpy as np
def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):
    input_length = int(math.log(max_int, 2))

    generator = Generator(input_length)
    discriminator = Discriminator(input_length)
    
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.01)

    loss_fn = torch.nn.BCELoss()
    
    for step in range(training_steps):
        generator_optimizer.zero_grad()

        noise = torch.randint(0, 2, (batch_size, input_length)).float()
        generated_data = generator(noise)

        true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
        true_labels = torch.tensor(true_labels).float().view(-1, 1)
        true_data = torch.tensor(true_data).float()

        generator_discriminator_output = discriminator(generated_data)
        generator_loss = loss_fn(generator_discriminator_output, true_labels)
        generator_loss.backward()
        generator_optimizer.step()

        discriminator_optimizer.zero_grad()
        true_discriminator_output = discriminator(true_data)
        true_discriminator_loss = loss_fn(true_discriminator_output, true_labels)

        generator_discriminator_output = discriminator(generated_data.detach())

        generator_discriminator_loss = loss_fn(generator_discriminator_output, torch.zeros_like(generator_discriminator_output))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        print(f"Step {step} - Generator Loss: {generator_loss.item()}, Discriminator Loss: {true_discriminator_loss.item()}")

    final_generated_data = generator(torch.randint(0, 2, (100, input_length)).float())
    print(np.round(final_generated_data.detach().numpy()))
        
def generate_even_data(max_int: int, batch_size: int):
    numbers = []
    labels = []
    input_length = int(math.log(max_int, 2))

    for i in range(batch_size):
        number = int(np.floor(np.random.randint(0, max_int/2)) * 2)
        number_bin = [int(x) for x in bin(number)[2:]]
        
        number_bin = [0] * (input_length - len(number_bin)) + number_bin
        
        numbers.append(number_bin)
        labels.append(number % 2 == 0)

    numbers = torch.tensor(numbers)
    labels = torch.tensor(labels)

    return labels, numbers
    
    
if __name__ == "__main__":
    train(max_int=8, batch_size=256, training_steps=10000)