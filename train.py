# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import wandb

from datetime import timedelta

import csv
import timm
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist

from tqdm import tqdm
# from torch.linalg import vector_norm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size


logger = logging.getLogger(__name__)


VIT_MODELS = {
    "ViT-T_16": "vit_tiny_patch16_224",
    "ViT-T_32": "vit_tiny_patch32_224",
    "ViT-S_16": "vit_small_patch16_224",
    "ViT-S_32": "vit_small_patch32_224",
    "ViT-B_16": "vit_base_patch16_224",
    "ViT-B_32": "vit_base_patch32_224",
    "ViT-L_16": "vit_large_patch16_224",
    "ViT-L_32": "vit_large_patch32_224",
    "ViT-H_14": "vit_huge_patch14_224",
    "R50-ViT-B_16": "vit_base_resnet50_224_in21k",
}

def save_metrics_to_csv(output_dir, name, global_step, train_loss, train_acc, val_loss, val_acc):
    csv_path = os.path.join(output_dir, f"{name}_metrics.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Step", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"])
        writer.writerow([global_step, train_loss, train_acc, val_loss, val_acc])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    num_classes = 10 if args.dataset == "cifar10" else 100
    # Prepare model
    config = CONFIGS[args.model_type]
    if args.model_type in ["ViT-T_16", "ViT-T_32", "ViT-S_16", "ViT-S_32"]:
        model_name = VIT_MODELS[args.model_type]    
        logger.info(f"Loading pretrained model {model_name} from timm...")
    
        timm_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smo=args.smo)
        model.load_from_timm(timm_model)
    else:
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smo=args.smo)
        model.load_from(np.load(args.pretrained_dir))
        
    model.to(args.device)
    num_params = count_parameters(model)
    
    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    if args.smo:
        logger.info(f"Training smo with alpha={args.alpha}, beta={args.beta}, lam={args.lam}")
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            if args.smo:
                logits, _ = model.pre(x)
            else:
                logits, _ = model(x)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy, eval_losses.avg

def vectorize(x, multichannel=False):
    """Vectorize data in any shape.

    Args:
        x (torch.Tensor): input data
        multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

    Returns:
        torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel: # one channel
            return x.reshape(x.shape[0], -1)
        else: # multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)

def loss_vi(x0, x, xp, coef, alpha=0.1, beta=0.1, lam=0):
    """INSO Loss"""
    # print(coef)
    loss_fct = torch.nn.CrossEntropyLoss()
    s1 = loss_fct(x, x0) / 2 + loss_fct(xp, x0) / 2
    s2 = alpha * coef.pow(2) - beta * coef.pow(2).log()
    r = vectorize(x).to(x.dtype)
    rp = vectorize(xp).to(x.dtype)
    s3 = (r - rp).pow(2).sum(dim=1).mean()
    # s3 = (vector_norm(r - rp, 2, dim=1)).pow(2).mean()
    # print(s1, s2, lam* s3)
    return s1 + s2 + lam * s3

def inject_noise(x, y, model, noise_type, noise_ratio, std_or_epsilon, device, visualize=False):
    """
    Applies noise to input tensor x based on the specified noise type.
    If visualize=True, returns both original and noisy images.
    """
    original_x = x.clone()  # Store original image for visualization
    
    if noise_type == "gaussian":
        noise = torch.randn_like(x) * std_or_epsilon  # Gaussian noise
        batch_mask = torch.rand(x.shape[0], device=x.device) < noise_ratio
        batch_mask = batch_mask.view(-1, 1, 1, 1) 
        x = x + noise * batch_mask
        x = x.clamp(0, 1)

    elif noise_type == "fgsm":
        batch_mask = torch.rand(x.shape[0], device=x.device) < noise_ratio
        if batch_mask.sum() == 0:
            return x
        
        x_selected = x[batch_mask].clone().requires_grad_(True)
        logits, _ = model(x_selected)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, y[batch_mask])
        model.zero_grad()
        loss.backward()
        grad = x_selected.grad.sign()
        x[batch_mask] = (x[batch_mask] + std_or_epsilon * grad).clamp(0, 1)

    if visualize:
        return original_x, x  # Return both original and noisy images
    return x

def visualize_images(original, noisy, step, output_dir):
    """
    Visualizes and saves a comparison of original and noisy images.
    """
    num_images = min(5, original.shape[0])  # Select 5 images to visualize
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))

    for i in range(num_images):
        # Convert tensors to numpy arrays
        orig_img = original[i].permute(1, 2, 0).cpu().numpy()
        noisy_img = noisy[i].permute(1, 2, 0).cpu().numpy()

        axes[0, i].imshow(orig_img)
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        axes[1, i].imshow(noisy_img)
        axes[1, i].axis("off")
        axes[1, i].set_title("Noisy")

    plt.suptitle(f"Step {step}: Noise Visualization")
    plt.savefig(os.path.join(output_dir, f"noise_visualization_step_{step}.png"))
    plt.show()


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
        wandb.init(project="vit-training", name=args.name, config=args)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    train_acc_metric = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            # Inject noise with visualization option enabled every N steps
            if global_step % args.eval_every == 0:
                original_x, x = inject_noise(x, y, model, args.noise, args.noise_ratio, args.std_or_epsilon, args.device, visualize=True)
                output_dir = args.output_dir + '/'  + str(args.noise) + str(args.noise_ratio) + '_' + str(args.std_or_epsilon)
                os.makedirs(output_dir, exist_ok=True)
                visualize_images(original_x, x, global_step, output_dir)
            else:
                x = inject_noise(x, y, model, args.noise, args.noise_ratio, args.std_or_epsilon, args.device)

            if args.smo:
                # Forward pass (INSO with stochastic perturbation)
                output1, _ = model(x)
                output2, _ = model(x)
    
                # Compute INSO loss
                loss = loss_vi(y, output1, output2, model.coef,
                               args.alpha, args.beta, args.lam)
                preds = torch.argmax((output1 + output2)/2, dim=-1)
            else:
                loss = model(x, y)
                logits, _ = model(x)      # TODO
                preds = torch.argmax(logits, dim=-1)

            acc = (preds == y).float().mean()
            train_acc_metric.update(acc.item(), x.size(0))
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()


            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                    wandb.log({
                            "train/loss": losses.val,
                            "train/lr": scheduler.get_lr()[0],
                            "train/accuracy": acc.item(),
                        }, step=global_step)


                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy, val_loss = valid(args, model, writer, test_loader, global_step)
                    save_metrics_to_csv(args.output_dir, args.name, global_step, 
                                      losses.avg, train_acc_metric.avg, val_loss, accuracy)
                    
                    wandb.log({
                        "val/loss": val_loss,
                        "val/accuracy": accuracy,
                        "train/avg_accuracy": train_acc_metric.avg,
                        "train/avg_loss": losses.avg
                    }, step=global_step)
    
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()
                    train_acc_metric.reset()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
        wandb.finish()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Smoothing parameters
    parser.add_argument("--smo", action='store_true',
                        help="Whether to use smoothing procedure")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=0)
    # Noise parameters
    parser.add_argument("--noise", choices=["none", "gaussian", "fgsm"], default="none")
    parser.add_argument("--noise_ratio", type=float, default=0)
    parser.add_argument("--std_or_epsilon", type=float, default=0)
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16",
                                                 "ViT-T_16", "ViT-T_32", "ViT-S_16", "ViT-S_32"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
