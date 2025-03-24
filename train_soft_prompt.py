import os

import numpy as np
import torch
import torch.nn.functional as F

from tokens import PAD
from setup import save_checkpoint, load_from_checkpoint
from board_ops import batch_detect_illegal_moves

save_interval = 1000

wandb_log = False
wandb_project = "ttt"

batch_size = 2048

learning_rate = 0.1
max_iters = 10000

device = "cuda"

data_dir = "data"
train_data = np.load(os.path.join(data_dir, "train.npy")).astype(dtype=np.int64)

def get_batch():
    data = train_data
    ix = torch.randint(data.shape[0], (batch_size,))
    x = torch.from_numpy(data[ix, :])
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = PAD
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
        device, non_blocking=True
    )
    return x, y

def calculate_loss(games, log_probs):
    batch_inds = torch.arange(log_probs.size(0))
    move_ind = 2
    probs = log_probs.exp()
    moves = torch.multinomial(probs[:, move_ind], 1).flatten()
    move_log_probs = log_probs[batch_inds, move_ind, moves]
    
    # loss = ((8 - moves) * move_log_probs).sum() * -1
    # loss = (moves * move_log_probs).sum() * -1
    loss = (moves * move_log_probs).sum()

    average_move = moves.sum() / moves.size(0)
    move_counts = torch.bincount(moves)
    return loss, average_move, move_counts


model = load_from_checkpoint()
model.to(device)
model.train()
model.zero_grad()
model.use_prompt = True

model.soft_prompt = torch.nn.Parameter(torch.randn(model.config.n_embd).to(device))
print(model.soft_prompt.sum())

for module in model.modules():
    module.requires_grad = False
model.soft_prompt.requires_grad = True
optimizer = torch.optim.Adam([model.soft_prompt], lr=learning_rate)

if wandb_log:
    import wandb

    wandb.init(project=wandb_project)

iter_num = 0
while iter_num < max_iters:
    X, _ = get_batch()

    logits, _ = model(X)
    log_probs = F.log_softmax(logits, dim=-1)
    loss, average_move, move_counts = calculate_loss(X, log_probs)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    lossf = loss.item()

    if iter_num > 0 and iter_num % save_interval == 0:
        save_checkpoint(model)
        
    if iter_num % 50 == 0:
        print(f"iter {iter_num}:\nloss {lossf:.4f}\naverage: {average_move}\ncounts: {move_counts}\nsanity: {model.soft_prompt.sum()}\n")

    if wandb_log:
        wandb.log(
            {
                "iter": iter_num,
                "train/loss": lossf,
                "lr": learning_rate,
            }
        )

    iter_num += 1
