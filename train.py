import os
import argparse
import yaml
from math import exp
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from model import Model
from datasets import TextDataset
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

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()

    return args


def train_epoch(model, data_loader, optimizer, scaler, scheduler, tb_writer, epoch, step, num_processed_samples, device,
                amp=False, num_accumulation_steps=1):

    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_samples = len(data_loader) * data_loader.batch_size
    with tqdm(total=num_samples, desc=f'Epoch {epoch}') as progress_bar:
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
            cur_batch_size = len(inputs)
            num_processed_samples += cur_batch_size
            tb_writer.add_scalar(tag='train loss', scalar_value=loss_value, global_step=num_processed_samples)
            progress_bar.update(cur_batch_size)
            progress_bar.set_postfix(step=step, lr=f'{cur_lr:.2e}', loss=f'{loss_value:.4f}')

            if i % num_accumulation_steps == 0:
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
    for inputs, targets, y in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with autocast(enabled=amp):
            logits = model(inputs=inputs)
            loss_sum += F.cross_entropy(input=logits.transpose(1, 2), target=targets,
                                        ignore_index=data_loader.dataset.pad_token_id, reduction='sum').item()
            num_tokens += (targets != data_loader.dataset.pad_token_id).sum().item()

    loss_mean = loss_sum / num_tokens
    perplexity = exp(loss_mean)

    print(f'Val perplexity: {perplexity:.4f}')
    tb_writer.add_scalar(tag='val perplexity', scalar_value=perplexity, global_step=num_processed_samples)

    return perplexity


def main():
    args = get_args()

    save_dir = get_save_dir(base_dir=args.save_dir, name=args.name)
    print(f'Results will be saved to {save_dir}.')

    with open(os.path.join(save_dir, 'args.yaml'), 'w') as file:
        yaml.safe_dump(vars(args), file, sort_keys=False)

    tb_writer = SummaryWriter(log_dir=save_dir)

    logger = Logger(save_dir=save_dir, metric='perplexity', maximize_metric=False)

    print('Preparing data...')
    train_dataset = TextDataset(os.path.join('data', f'{args.dataset}_train.txt'))
    val_dataset = TextDataset(os.path.join('data', f'{args.dataset}_val.txt'))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn,
                              shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate_fn,
                            shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    num_steps = len(train_loader) // args.num_accumulation_steps * args.num_epochs

    print('Creating model...')
    model = Model(num_layers=args.num_layers,
                  hidden_dim=args.hidden_dim,
                  num_heads=args.num_heads,
                  hidden_dim_multiplier=args.hidden_dim_multiplier,
                  num_token_embeddings=train_dataset.tokenizer.vocab_size,
                  num_pos_embeddings=512,
                  dropout=args.dropout,
                  attn_dropout=args.attn_dropout)

    if args.pretrained_model is not None:
        state_dict = torch.load(args.pretrained_model)

        del state_dict['output_linear.weight']
        del state_dict['output_linear.bias']

        model.load_state_dict(state_dict, strict=False)

    model.to(args.device)

    parameter_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    scheduler = get_lr_scheduler(optimizer=optimizer, num_steps=num_steps, num_warmup_steps=args.num_warmup_steps,
                                 warmup_proportion=args.warmup_proportion)

    print('Starting training...')
    best_val_metric = None
    best_epoch = None
    step = 1
    num_processed_samples = 0
    for epoch in range(1, args.num_epochs + 1):
        step, num_processed_samples = train_epoch(model=model, data_loader=train_loader, optimizer=optimizer,
                                                  scaler=scaler, scheduler=scheduler, tb_writer=tb_writer,
                                                  epoch=epoch, step=step, num_processed_samples=num_processed_samples,
                                                  device=args.device, amp=args.amp,
                                                  num_accumulation_steps=args.num_accumulation_steps)

        val_metric = evaluate(model=model, data_loader=val_loader, tb_writer=tb_writer,
                              num_processed_samples=num_processed_samples, device=args.device, amp=args.amp)

        print()

        logger.update_metrics(val_metric=val_metric, epoch=epoch)

        if best_val_metric is None or val_metric < best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    print(f'Best val perplexity: {best_val_metric} (achieved after epoch {best_epoch})')
    print()


if __name__ == '__main__':
    main()
