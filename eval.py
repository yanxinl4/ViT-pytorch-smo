# eval.py
import argparse
import os
import logging
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.modeling import VisionTransformer, CONFIGS
from models.tiny_vit import tiny_vit_5m_224, tiny_vit_11m_224, tiny_vit_21m_224
from utils.data_utils import get_loader


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def evaluate(args, model, test_loader):
    model.eval()
    eval_losses = AverageMeter()
    all_preds, all_labels = [], []
    loss_fct = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            
            if args.model_type in ["TinyViT-5M", "TinyViT-11M", "TinyViT-21M"]:
                logits = model(x)
            else:
                logits, _ = model(x)
            
            loss = loss_fct(logits, y)
            eval_losses.update(loss.item())

            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # 计算指标
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = (all_preds == all_labels).mean()
    
    print(f"\nEvaluation Results:")
    print(f"Test Loss: {eval_losses.avg:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return accuracy, eval_losses.avg

class AverageMeter:
    """Computes and stores average and current value"""
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

def load_model(args):
    num_classes = 10 if args.dataset == "cifar10" else 100
    
    if args.model_type in ["TinyViT-5M", "TinyViT-11M", "TinyViT-21M"]:
        model_classes = {
            "TinyViT-5M": tiny_vit_5m_224,
            "TinyViT-11M": tiny_vit_11m_224,
            "TinyViT-21M": tiny_vit_21m_224
        }
        model = model_classes[args.model_type](
            pretrained=False, 
            num_classes=num_classes,
            img_size=args.img_size
        )
    else:
        config = CONFIGS[args.model_type]
        model = VisionTransformer(
            config, 
            args.img_size, 
            num_classes=num_classes, 
            zero_head=True,
            smo=args.smo
        )
    
    if args.checkpoint_path.endswith(".npz"):
        model.load_from(np.load(args.checkpoint_path))
    else:  # PyTorch checkpoint
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        if "module." in list(state_dict.keys())[0]:  
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        if 'coef' in state_dict:
            coef = state_dict['coef']
            if coef.dim() == 0:
                state_dict['coef'] = coef.unsqueeze(0)
        model.load_state_dict(state_dict, strict=False)
    
    model.to(args.device)
    print(model.coef)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smo", action='store_true',
                        help="Whether to use smoothing procedure")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained checkpoint")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                "ViT-L_32","ViT-S_16", "ViT-T_16",
                                                "TinyViT-5M", "TinyViT-11M",
                                                "TinyViT-21M"], required=True,
                        help="Model architecture")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--save_file", default="")
    
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Evaluation seed")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    
    
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps. Will always run one evaluation at the end of training.")

    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.seed != -1:
        set_seed(args.seed)
    
    
    logger.info("Loading model...")
    model = load_model(args)
    
    
    logger.info("Loading data...")
    _, test_loader = get_loader(args)
    
    
    logger.info("Starting evaluation...")
    accuracy, loss = evaluate(args, model, test_loader)
    
    
    result_str = f"Results of {args.checkpoint_path} with seed: {args.seed}\nAccuracy: {accuracy*100:.4f}%\nLoss: {loss:.4f}\n\n"
    with open(f"{args.save_file}_eval_results.txt", "a") as f:
        f.write(result_str)
    print(result_str)

if __name__ == "__main__":
    main()
    