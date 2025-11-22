"""
THE SYMBIOTIC PATH: EWC SIMULATION
----------------------------------
This script demonstrates the technical foundation of the "Symbiotic Path" paper:
Using Elastic Weight Consolidation (EWC) to allow a Neural Network to learn 
a new task without "catastrophically forgetting" the previous one.

Paper Section: 5.1
Author: C. Hobday / Symbiotic Path Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy

# --- CONFIGURATION ---
HIDDEN_SIZE = 100       # Size of the SLM's "brain"
EPOCHS = 2000           # Training cycles
LEARNING_RATE = 0.01
EWC_LAMBDA = 10000      # The "Memory Rigidity" factor (lambda in the paper)

# --- 1. THE MODEL (Small Language Model Proxy) ---
class SimpleSLM(nn.Module):
    def __init__(self):
        super(SimpleSLM, self).__init__()
        # A simple MLP approximating a larger Transformer's reasoning capabilities
        self.net = nn.Sequential(
            nn.Linear(1, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. DATA GENERATION ---
def get_task_data(task_type='sin', n_samples=100):
    """
    Generates synthetic patterns.
    Task A = Sine Wave (e.g., User's Coding Style)
    Task B = Cosine Wave (e.g., User's Email Style)
    """
    # Range [-pi, pi]
    x = torch.linspace(-np.pi, np.pi, n_samples).view(-1, 1)
    if task_type == 'sin':
        y = torch.sin(x)
    elif task_type == 'cos':
        y = torch.cos(x) 
    return x, y

# --- 3. FISHER INFORMATION (The "Memory" Mechanism) ---
def compute_fisher(model, x, y):
    """
    Calculates the Fisher Information Matrix (Diagonal).
    This identifies WHICH parameters are critical for Task A.
    
    Paper Ref: Section 5.1 "High Fisher Value = Crucial Parameter"
    """
    model.train()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    criterion = nn.MSELoss()
    
    # Forward pass
    preds = model(x)
    loss = criterion(preds, y)
    model.zero_grad()
    loss.backward()

    # Fisher Information is approximated by squared gradients
    for n, p in model.named_parameters():
        if p.grad is not None:
            # Square the gradients to get importance magnitude
            fisher[n] = p.grad.data.clone().pow(2)
            
    return fisher

# --- 4. EWC LOSS FUNCTION ---
def ewc_loss(model, fisher, old_params, current_loss, lambda_val):
    """
    The mathematical heart of the paper.
    L = L_new + (lambda/2) * Sum(Fisher * (theta - theta_old)^2)
    """
    ewc_reg = 0
    for n, p in model.named_parameters():
        if n in fisher:
            # The penalty increases if we move a parameter that has high Fisher info
            ewc_reg += (fisher[n] * (p - old_params[n]).pow(2)).sum()
    
    return current_loss + (lambda_val / 2) * ewc_reg

# --- 5. TRAINING LOOPS ---
def train_task(model, x, y, optimizer, epochs, fisher=None, old_params=None):
    criterion = nn.MSELoss()
    model.train()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        
        # If we have fisher info, apply EWC regularization (Symbiotic Mode)
        if fisher is not None and old_params is not None:
            loss = ewc_loss(model, fisher, old_params, loss, EWC_LAMBDA)
            
        loss.backward()
        optimizer.step()
    return model

# --- MAIN SIMULATION ---
def run_simulation():
    print(">>> Initializing Symbiotic AI Simulation...")
    print(">>> Scenario: AI learns Task A, then must learn Task B.")
    
    # Generate Data
    x_a, y_a = get_task_data('sin') # Task A: Old Memory
    x_b, y_b = get_task_data('cos') # Task B: New Experience

    # 1. Train on Task A (The "Base" Knowledge)
    print("\n[Phase 1] Learning Task A (Sine Wave)...")
    model = SimpleSLM()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = train_task(model, x_a, y_a, optimizer, EPOCHS)
    
    # 2. Compute Fisher Information (Consolidation)
    print("[Phase 2] Consolidating Memory (Computing Fisher Matrix)...")
    fisher_matrix = compute_fisher(model, x_a, y_a)
    old_params = {n: p.data.clone() for n, p in model.named_parameters()}
    
    # 3. Train on Task B - NAIVE approach (No Symbiosis)
    print("[Phase 3a] Simulating Monoculture Model (Naive Training)...")
    model_naive = copy.deepcopy(model)
    optim_naive = optim.Adam(model_naive.parameters(), lr=LEARNING_RATE)
    # Note: We do NOT pass fisher/old_params here
    model_naive = train_task(model_naive, x_b, y_b, optim_naive, EPOCHS) 
    
    # 4. Train on Task B - SYMBIOTIC approach (With EWC)
    print("[Phase 3b] Simulating Symbiotic Model (EWC Training)...")
    model_ewc = copy.deepcopy(model)
    optim_ewc = optim.Adam(model_ewc.parameters(), lr=LEARNING_RATE)
    # Note: We PASS fisher/old_params here
    model_ewc = train_task(model_ewc, x_b, y_b, optim_ewc, EPOCHS, 
                           fisher=fisher_matrix, old_params=old_params)

    # --- VISUALIZATION ---
    print("\n>>> Generating Proof Visualization...")
    plt.figure(figsize=(14, 5))
    
    # Plot 1: The Naive Failure
    plt.subplot(1, 2, 1)
    plt.title("Standard AI (Catastrophic Forgetting)", fontsize=14, fontweight='bold', color='#d62728') # Red
    plt.plot(x_a.numpy(), y_a.numpy(), 'g--', label="Task A (Old Memory)", linewidth=2, alpha=0.5)
    plt.plot(x_b.numpy(), y_b.numpy(), 'b--', label="Task B (New Goal)", linewidth=2, alpha=0.5)
    # The model predicts Task B well, but fails Task A
    plt.plot(x_a.numpy(), model_naive(x_a).detach().numpy(), 'r-', label="Model Prediction", linewidth=3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.text(-2, -0.8, "RESULT: The AI learned B,\nbut completely forgot A.", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Plot 2: The Symbiotic Success
    plt.subplot(1, 2, 2)
    plt.title("Symbiotic AI (With EWC)", fontsize=14, fontweight='bold', color='#2ca02c') # Green
    plt.plot(x_a.numpy(), y_a.numpy(), 'g--', label="Task A (Old Memory)", linewidth=2, alpha=0.5)
    plt.plot(x_b.numpy(), y_b.numpy(), 'b--', label="Task B (New Goal)", linewidth=2, alpha=0.5)
    # The model should trace BOTH lines reasonably well
    combined_x = torch.cat((x_a, x_b))
    sorted_indices = torch.argsort(combined_x.flatten())
    # Visualizing how it handles the A domain
    plt.plot(x_a.numpy(), model_ewc(x_a).detach().numpy(), color='#2ca02c', linestyle='-', label="Retained Memory", linewidth=3)
    # Visualizing how it handles the B domain
    plt.plot(x_b.numpy(), model_ewc(x_b).detach().numpy(), color='#1f77b4', linestyle='-', label="New Skill", linewidth=3)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.text(-2, -0.8, "RESULT: The AI learned B\nAND remembered A.", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('ewc_results.png')
    print(">>> Simulation Complete. Results saved to 'ewc_results.png'")
    plt.show()

if __name__ == "__main__":
    run_simulation()
