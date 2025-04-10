import os
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F

from tokens import PAD
from setup import save_checkpoint, load_from_checkpoint
from board_ops import batch_seq_to_board, optimal_moves, batch_legal_moves_from_seq, batch_check_winner, seq_move_to_board_move, win_or_block_moves

save_interval = 1000

wandb_log = False
wandb_project = "ttt"

max_iters = 10000

# batch_size = 2048
# learning_rate = 0.015
# entropy_coef = 0.8

batch_size = 2048
learning_rate = 0.3
entropy_coef = 0.2

class MoveValue(Enum):
    OPTIMAL = 0
    SUBOPTIMAL = -1
    ILLEGAL = -2
    

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

def batch_sample_moves_for_seq_pos(log_probs, seq_pos, temperature=1.0, k=None):
    log_probs = log_probs[:, seq_pos, :] / temperature
    if k is not None:
        v, _ = torch.topk(log_probs, k)
        log_probs[log_probs < v[:, [-1]]] = -float("Inf")
    probs = log_probs.exp()
    dist = torch.distributions.Categorical(logits=log_probs)
    entropy = dist.entropy()
    moves = dist.sample()
    return moves, entropy


model = load_from_checkpoint()
model.to(device)
model.train()
model.zero_grad()
model.use_prompt = True

# model.soft_prompt = torch.nn.Parameter(torch.randn(model.config.n_embd).to(device))
model.soft_prompt = torch.nn.Parameter(model.transformer.wte(torch.tensor(10).to(device)))
print(f"{model.soft_prompt.sum()=}")

for module in model.modules():
    module.requires_grad = False
model.soft_prompt.requires_grad = True
optimizer = torch.optim.Adam([model.soft_prompt], lr=learning_rate)

if wandb_log:
    import wandb

    wandb.init(project=wandb_project)

iter_num = 0
while iter_num < max_iters:
    batch_seq, _ = get_batch()    

    logits, _ = model(batch_seq)
    log_probs = F.log_softmax(logits, dim=-1)

    loss = 0
    ignored_count = 0
    illegal_count = 0
    suboptimal_count = 0
    optimal_count = 0
    total_entropy = 0
    pos_counts = [0] * 10
    move_counts = [0] * 11

    player = -1
    for i in range(2, batch_seq.size(1)):
        batch_subseq = batch_seq[:, :i]
        batch_boards = batch_seq_to_board(batch_subseq)
        winners = batch_check_winner(batch_boards.cpu())
        legal_moves = batch_legal_moves_from_seq(batch_subseq)
        played_moves, entropy = batch_sample_moves_for_seq_pos(log_probs, i-1, temperature=1.0, k=None)

        total_entropy += entropy.sum()

        for j in range(batch_seq.size(0)):
            if winners[j] != 0:
                ignored_count += 1
                continue

            move = played_moves[j]
            move_counts[move.item()] += 1

            for k in range(log_probs.size(-1)):
                loss += (log_probs[j][i][k] * k * -1)
            # loss += (log_probs[j][i][move] * (-1 * (move != 3)))
            continue
            

            if move.item() not in legal_moves[j]:
                loss += (log_probs[j][i][move] * MoveValue.ILLEGAL.value - (entropy[j] * entropy_coef))
                illegal_count += 1

                # print(f"Illegal seq pos: {i}")
                # print(batch_boards[j])
                # print(move)
                # print(log_probs[j][i].exp())
                # print(batch_subseq[j])
                # print(batch_seq[j])
                # print("\n")
                
                pos_counts[i] += 1
                move_counts[move.item()] += 1

            else:
                # loss += (log_probs[j][i][move] * (min(legal_moves[j]) - move) - (entropy[j] * entropy_coef))
                
                # best_board_moves = optimal_moves(batch_boards[j].cpu(), player)
                best_board_moves = win_or_block_moves(batch_boards[j].cpu(), player)
                if best_board_moves is not None and seq_move_to_board_move(move) not in best_board_moves:
                    loss += (log_probs[j][i][move] * MoveValue.SUBOPTIMAL.value - (entropy[j] * entropy_coef))
                    suboptimal_count += 1
                else:
                    optimal_count += 1

        player *= -1

    # print(f"{pos_counts=}")
    print(f"{move_counts=}")
    # print(f"{sum(pos_counts)=}")
    # print("\n")
    # print("\n")

    loss.backward()

    
    if iter_num > 0 and iter_num % save_interval == 0:
        save_checkpoint(model)
        
    if iter_num % 1 == 0:
        lossf = loss.item()
        print(f"iter {iter_num}:\nloss: {lossf:.4f}\nentropy: {total_entropy:.4f}\nsanity: {model.soft_prompt.sum():.4f}")
        print(f"ignored: {ignored_count}\nillegal: {illegal_count}\nsuboptimal: {suboptimal_count}\noptimal: {optimal_count}\n")

        
    if wandb_log:
        wandb.log(
            {
                "iter": iter_num,
                "train/loss": lossf,
                "lr": learning_rate,
            }
        )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    iter_num += 1
