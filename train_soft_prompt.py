import os
import time

import numpy as np
import torch

from tokens import PAD
from setup import init_model, save_checkpoint
import torch.nn.functional as F

save_interval = 1000

wandb_log = False
wandb_project = "ttt"

batch_size = 2048

learning_rate = 6e-4
max_iters = 600000

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

def calculate_reward(token_ids):
    print(token_ids)
    print(token_ids.shape)
    return


model = init_model()
model.to(device)
model.train()
model.zero_grad()
model.use_soft_prompt = True

for module in model.modules():
    module.requires_grad = False
model.soft_prompt.requires_grad = True
optimizer = torch.optim.Adam([model.soft_prompt], lr=learning_rate)

if wandb_log:
    import wandb

    wandb.init(project=wandb_project)

X, Y = get_batch()
t0 = time.time()
iter_num = 0
while iter_num < max_iters:
    if iter_num > 0 and iter_num % save_interval == 0:
        save_checkpoint(model)

    logits, _ = model(X, Y)

    log_probs = F.log_softmax(logits[:, -1], dim=-1)
    probs = log_probs.exp()
    token_ids = torch.multinomial(probs, 1)

    reward = calculate_reward(token_ids)
    break

    loss.backward()

    X, Y = get_batch()


    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    lossf = loss.item()

    if wandb_log:
        wandb.log(
            {
                "iter": iter_num,
                "train/loss": lossf,
                "lr": learning_rate,
            }
        )

    iter_num += 1
