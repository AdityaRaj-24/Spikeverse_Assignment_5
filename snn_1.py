# Spiking Neural Network for MNIST using Poisson Rate Coding and LIF Neurons

# --- Visualizations ---
def visualize_spike_encoding(images, T=100):
    """Plot Poisson spike raster for one example from the batch."""
    x = images[0].view(-1)  # Flatten first image
    spikes = poisson_encode(x.unsqueeze(0), T, f_max)[0].cpu().numpy()  # [784, T]
    plt.figure(figsize=(10, 6))
    for i in range(784):
        spike_times = np.where(spikes[i] == 1)[0]
        plt.scatter(spike_times, np.full_like(spike_times, i), s=1, color='black')
    plt.xlabel("Time (ms)")
    plt.ylabel("Pixel Index")
    plt.title("Poisson Spike Raster for a Single MNIST Image")
    plt.show()

train_losses = []  # Global variable to store training loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 100  # simulation time (ms)
dt = 1   # time step (ms)
f_max = 100  # max firing rate in Hz
threshold = 1.0
reset_voltage = 0.0

# --- Poisson Rate Encoding ---
def poisson_encode(x, T, f_max):
    """Encodes normalized input into Poisson spike trains."""
    batch_size, input_dim = x.shape
    spike_prob = x.unsqueeze(2).expand(batch_size, input_dim, T)
    rand_vals = torch.rand_like(spike_prob)
    return (rand_vals < (spike_prob * f_max * dt / 1000)).float()

# --- Surrogate Gradient for Spikes ---
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        surrogate_grad = 1 / (1 + 10 * torch.abs(input - threshold)) ** 2
        return grad_input * surrogate_grad

# --- Leaky Integrate-and-Fire (LIF) Neuron ---
class LIFNeuronLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size) * 0.1)

    def forward(self, spike_input):
        batch_size, input_size, T = spike_input.shape
        output_size = self.weight.shape[1]

        membrane = torch.zeros((batch_size, output_size), device=device)
        spike_output = torch.zeros((batch_size, output_size, T), device=device)

        for t in range(T):
            input_t = spike_input[:, :, t]
            input_current = torch.matmul(input_t, self.weight)
            membrane += input_current
            fired = SurrogateSpike.apply(membrane)
            spike_output[:, :, t] = fired
            membrane = membrane * (1 - fired) + reset_voltage * fired

        return spike_output

# --- SNN Model ---
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = LIFNeuronLayer(28*28, 100)
        self.output = LIFNeuronLayer(100, 10)

    def forward(self, x):
        spikes = poisson_encode(x, T, f_max)
        hidden_spikes = self.hidden(spikes)
        output_spikes = self.output(hidden_spikes)
        return output_spikes

# --- Dataset Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# --- Training ---
def train(model, epochs=1):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    global train_losses
    train_losses = []

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            output_spikes = model(images)
            spike_counts = output_spikes.sum(dim=2)
            loss = F.cross_entropy(spike_counts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

# --- Evaluation ---
def evaluate(model):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output_spikes = model(images)
            spike_counts = output_spikes.sum(dim=2)
            predicted = spike_counts.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

# --- Training & Visualization ---
model = SNN().to(device)
dataiter = iter(train_loader)
sample_images, _ = next(dataiter)
visualize_spike_encoding(sample_images, T)
train(model, epochs=1)
evaluate(model)

# --- Plot Training Loss ---  
plt.plot(train_losses)
plt.xlabel("Batch Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()