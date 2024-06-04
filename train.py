import os
import sys
import time
import math
import argparse
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from model import ModelConfig, RNN, loss_fn
from data_loader import CharDataset, create_datasets, InfiniteDataLoader


# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def print_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-'*80)

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits = model(X)
        loss = loss_fn(logits, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss


if __name__ == '__main__':
    
    # I/O
    parser = argparse.ArgumentParser(description="Make More")
    
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='data/names.txt', 
                        help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', 
                        help="output working directory")
    parser.add_argument('--resume', action='store_true', 
                        help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', 
                        help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=0, # original default valu is 4, but gives error
                        help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, 
                        help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', 
                        help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407,
                        help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, 
                        help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='rnn', 
                        help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, 
                        help="number of layers")
    parser.add_argument('--n-embd', type=int, default=64, 
                        help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, 
                        help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, 
                        help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, 
                        help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, 
                        help="weight decay")

    args = parser.parse_args()
    print(vars(args))


    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    #writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                        n_layer=args.n_layer, 
                        n_embd=args.n_embd, 
                        n_embd2=args.n_embd2
                        )
    model = RNN(config, cell_type=args.type)
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()
        
    # init optimizer
    print('Initializing optimizer')
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=args.learning_rate, 
                                weight_decay=args.weight_decay,
                                betas=(0.9, 0.99),
                                eps=1e-8)

    # init dataloader
    print('Initializing dataloader')
    batch_loader = InfiniteDataLoader(train_dataset, 
                                    batch_size=args.batch_size, 
                                    pin_memory=True, 
                                    num_workers=args.num_workers)

    # -----------------------------------------------------------------------------
    # training loop
    best_loss = None
    step = 0
    print('Launch training loop')
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # feed into the model and calculate the loss
        logits = model(X)
        loss = loss_fn(logits, Y)
        
        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)

            print(f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss:.4f} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break

