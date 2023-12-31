import os
import argparse
import yaml
from math import exp
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from models import Transformer, CBoETransformer
from dataset import TextDataset
from utils import Logger, get_save_dir, get_parameter_groups, get_lr_scheduler


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, help='Experiment name.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, required=True)

    # model architecture
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=4)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--save_model', default=False, action='store_true')

    # CBoE
    parser.add_argument('--cboe', default=False, action='store_true')
    parser.add_argument('--cboe_every_layers', type=int, default=4)

    # regularization
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--attn_dropout', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)

    # training parameters
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Only used if num_warmup_steps is None.')

    parser.add_argument('--device_ids', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()

    return args


def train_epoch(model, data_loader, optimizer, scaler, scheduler, tb_writer, epoch, step, num_processed_samples, device,
                amp=False, num_accumulation_steps=1, rank=0, world_size=1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_samples = len(data_loader) * data_loader.batch_size * world_size
    with tqdm(total=num_samples, desc=f'Epoch {epoch}', disable=(rank != 0)) as progress_bar:
        for i, (inputs, targets) in enumerate(data_loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(enabled=amp):
                logits = model(inputs=inputs)
                loss = F.cross_entropy(input=logits.transpose(1, 2), target=targets,
                                       ignore_index=data_loader.dataset.pad_token_id)
                loss /= num_accumulation_steps

            scaler.scale(loss).backward()

            loss_value = loss.item() * num_accumulation_steps
            cur_lr = scheduler.get_last_lr()[0]
            cur_batch_size = len(inputs) * world_size
            num_processed_samples += cur_batch_size

            if tb_writer is not None:
                tb_writer.add_scalar(tag='train loss', scalar_value=loss_value, global_step=num_processed_samples)

            progress_bar.update(cur_batch_size)
            progress_bar.set_postfix(step=step, lr=f'{cur_lr:.2e}', loss=f'{loss_value:.4f}')

            if i % num_accumulation_steps == 0:
                if tb_writer is not None:
                    tb_writer.add_scalar(tag='lr', scalar_value=cur_lr, global_step=num_processed_samples)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

    return step, num_processed_samples


@torch.no_grad()
def evaluate(model, data_loader, tb_writer, num_processed_samples, device, amp=False):
    print('Evaluating...')
    model.eval()
    loss_sum = 0
    num_tokens = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with autocast(enabled=amp):
            logits = model(inputs=inputs)
            loss_sum += F.cross_entropy(input=logits.transpose(1, 2), target=targets,
                                        ignore_index=data_loader.dataset.pad_token_id, reduction='sum').item()
            num_tokens += (targets != data_loader.dataset.pad_token_id).sum().item()

    loss_mean = loss_sum / num_tokens
    perplexity = exp(loss_mean)

    print(f'Val perplexity: {perplexity:.4f}\n')
    if tb_writer is not None:
        tb_writer.add_scalar(tag='val perplexity', scalar_value=perplexity, global_step=num_processed_samples)

    return perplexity


def main():
    args = get_args()

    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        save_dir = get_save_dir(base_dir=args.save_dir, name=args.name)
        print(f'Results will be saved to {save_dir}.')

        with open(os.path.join(save_dir, 'args.yaml'), 'w') as file:
            yaml.safe_dump(vars(args), file, sort_keys=False)

        logger = Logger(save_dir=save_dir, metric='perplexity', maximize_metric=False)

        tb_writer = SummaryWriter(log_dir=save_dir)

    else:
        tb_writer = None

    if rank == 0:
        print('Preparing data...')

    train_dataset = TextDataset(os.path.join('data', f'{args.dataset}_train.txt.gz'))
    val_dataset = TextDataset(os.path.join('data', f'{args.dataset}_val.txt.gz'))
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                       drop_last=True)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              collate_fn=train_dataset.collate_fn, drop_last=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate_fn,
                            shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    num_steps = len(train_loader) // args.num_accumulation_steps * args.num_epochs

    if rank == 0:
        print('Creating model...')

    if args.cboe:
        model = CBoETransformer(num_layers=args.num_layers,
                                cboe_every_layers=args.cboe_every_layers,
                                hidden_dim=args.hidden_dim,
                                num_heads=args.num_heads,
                                hidden_dim_multiplier=args.hidden_dim_multiplier,
                                num_token_embeddings=train_dataset.tokenizer.vocab_size,
                                num_pos_embeddings=512,
                                dropout=args.dropout,
                                attn_dropout=args.attn_dropout)
    else:
        model = Transformer(num_layers=args.num_layers,
                            hidden_dim=args.hidden_dim,
                            num_heads=args.num_heads,
                            hidden_dim_multiplier=args.hidden_dim_multiplier,
                            num_token_embeddings=train_dataset.tokenizer.vocab_size,
                            num_pos_embeddings=512,
                            dropout=args.dropout,
                            attn_dropout=args.attn_dropout)

    if args.pretrained_model is not None:
        state_dict = torch.load(args.pretrained_model)
        consume_prefix_in_state_dict_if_present(state_dict)

        del state_dict['output_linear.weight']
        del state_dict['output_linear.bias']

        model.load_state_dict(state_dict, strict=False)

    device_id = int(args.device_ids.split(',')[rank])
    device = f'cuda:{device_id}'
    model.to(device)

    model = DDP(model, device_ids=[device_id], output_device=device_id)

    parameter_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    scheduler = get_lr_scheduler(optimizer=optimizer, num_steps=num_steps, num_warmup_steps=args.num_warmup_steps,
                                 warmup_proportion=args.warmup_proportion)

    if rank == 0:
        print('Starting training...')
        best_perplexity = None
        best_epoch = None

    step = 1
    num_processed_samples = 0
    for epoch in range(1, args.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        step, num_processed_samples = train_epoch(model=model, data_loader=train_loader, optimizer=optimizer,
                                                  scaler=scaler, scheduler=scheduler, tb_writer=tb_writer,
                                                  epoch=epoch, step=step, num_processed_samples=num_processed_samples,
                                                  device=device, amp=args.amp,
                                                  num_accumulation_steps=args.num_accumulation_steps,
                                                  rank=rank, world_size=world_size)

        if rank == 0:
            perplexity = evaluate(model=model, data_loader=val_loader, tb_writer=tb_writer,
                                  num_processed_samples=num_processed_samples, device=device, amp=args.amp)

            logger.update_metrics(val_metric=perplexity, epoch=epoch)

            if best_perplexity is None or perplexity < best_perplexity:
                best_perplexity = perplexity
                best_epoch = epoch
                if args.save_model:
                    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

        dist.barrier()

    if rank == 0:
        print(f'Best val perplexity: {best_perplexity} (achieved after epoch {best_epoch})\n')


if __name__ == '__main__':
    main()
