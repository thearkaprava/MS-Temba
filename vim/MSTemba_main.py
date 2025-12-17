import time
import argparse
import csv
from torch.autograd import Variable
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from utils import *
from apmeter import APMeter
import os
from torch.utils.tensorboard import SummaryWriter
import logging

# Import necessary modules from main_no_teacher.py
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import models_MSTemba

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str, default='True', help='train or eval')
parser.add_argument('-backbone', type=str, default='i3d')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-gpu', type=str, default='4')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default='/path/to/charades.json')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
# parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epochs', type=int, default=50)
parser.add_argument('-model', type=str, default='')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-batch_size', type=str, default='False')
parser.add_argument('-num_clips', type=str, default='False')
parser.add_argument('-skip', type=str, default='False')
parser.add_argument('-num_layer', type=str, default='False')
parser.add_argument('-unisize', type=str, default='False')
parser.add_argument('-alpha_l', type=float, default='1.0')
parser.add_argument('-beta_l', type=float, default='1.0')
parser.add_argument('-output_dir', type=str, default='./output', help='Directory to save output files')

# Add new arguments from main_no_teacher.py
parser.add_argument('--model', default='vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2', type=str, metavar='MODEL',
                    help='Name of model to train')
# parser.add_argument('--input-size', default=224, type=int, help='images input size')
# parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
# parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
parser.add_argument('--model-ema', action='store_true')
parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
parser.set_defaults(model_ema=True)
parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                    help='Drop path rate (default: 0.0)')
# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.01,
                    help='weight decay (default: 0.01)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

args = parser.parse_args()

# set random seed
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED:', SEED)

batch_size = int(args.batch_size)

from charades_dataloader import Charades as Dataset

def load_data(train_split, val_split, root):
    # Load Data
    print('load data', root)

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size, classes, int(args.num_clips), int(args.skip))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size, classes, int(args.num_clips), int(args.skip))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root
    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    
    return dataloaders, datasets


def run(models, criterion, num_epochs=50):
    since = time.time()
    Best_val_map = 0.
    Best_sample_val_map = 0.
    Best_block_sample_val_maps = [0., 0., 0.]  # One for each block
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))
    
    for epoch in range(num_epochs):
        since1 = time.time()
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            # Training step with block metrics
            train_map, train_loss, block_train_maps, avg_diversity_loss = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            logging.info(f'Epoch {epoch} - Train MAP: {train_map:.2f}, Train Loss: {train_loss:.4f}')
            
            # Validation step with block metrics
            prob_val, val_loss, val_map, sample_val_map, block_val_maps, block_sample_val_maps = val_step(model, gpu, dataloader['val'], epoch)
            logging.info(f'Epoch {epoch} - Val MAP: {val_map:.2f}, Val Loss: {val_loss:.4f}')
            logging.info(f'Epoch {epoch} - Sampled Val MAP: {sample_val_map:.2f}')
            sched.step(val_loss)
            
            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Loss/diversity', avg_diversity_loss, epoch)
            writer.add_scalar('mAP/train', train_map, epoch)
            writer.add_scalar('mAP/val', val_map, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Log block metrics to TensorBoard
            for i in range(3):
                writer.add_scalar(f'Block_{i+1}/train_map', block_train_maps[i], epoch)
                writer.add_scalar(f'Block_{i+1}/val_map', block_val_maps[i], epoch)
                writer.add_scalar(f'Block_{i+1}/sampled_val_map', block_sample_val_maps[i], epoch)
            
            # Time
            epoch_time = time.time() - since1
            total_time = time.time() - since
            logging.info(f"Epoch {epoch}, Total_Time {total_time:.2f}, Epoch_time {epoch_time:.2f}")
            writer.add_scalar('Time/epoch', epoch_time, epoch)
            writer.add_scalar('Time/total', total_time, epoch)

            # # Save best models based on sample_val_map
            # if Best_sample_val_map < sample_val_map:
            #     Best_sample_val_map = sample_val_map
            #     logging.info(f"Epoch {epoch}, Best Sampled Val Map Update {Best_sample_val_map:.4f}")
            #     pickle.dump(prob_val, open(os.path.join(args.output_dir, f'{epoch}.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
            #     logging.info(f"Logit saved at: {args.output_dir}/{epoch}.pkl")
                
            #     # Save best model
            #     torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            #     logging.info(f"Best model saved at: {args.output_dir}/best_model.pth")
            #     writer.add_scalar('Best_mAP/val', Best_val_map, epoch)

            # Save best models based on val_map
            if Best_val_map < val_map:
                Best_val_map = val_map
                logging.info(f"Epoch {epoch}, Best Sampled Val Map Update {Best_val_map:.4f}")
                pickle.dump(prob_val, open(os.path.join(args.output_dir, f'{epoch}.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
                logging.info(f"Logit saved at: {args.output_dir}/{epoch}.pkl")
                
                # Save best model
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                logging.info(f"Best model saved at: {args.output_dir}/best_model.pth")
                writer.add_scalar('Best_mAP/val', Best_val_map, epoch)
            
            # Save best models for each block
            for i in range(3):
                if Best_block_sample_val_maps[i] < block_sample_val_maps[i]:
                    Best_block_sample_val_maps[i] = block_sample_val_maps[i]
                    logging.info(f"Epoch {epoch}, Block {i+1} Best Sampled Val Map Update {Best_block_sample_val_maps[i]:.4f}")
                    block_dir = os.path.join(args.output_dir, f'block_{i+1}')
                    torch.save(model.state_dict(), os.path.join(block_dir, 'best_model.pth'))
                    logging.info(f"Block {i+1} Best model saved at: {block_dir}/best_model.pth")
                    writer.add_scalar(f'Block_{i+1}/Best_sampled_val_map', Best_block_sample_val_maps[i], epoch)
    
    writer.close()


def eval_model(model, dataloader, baseline=False):
    results = {}
    block_results = [{} for _ in range(3)]  # One dict for each block
    
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _, block_probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        # Store final output results
        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
        
        # Store block results
        for i, block_prob in enumerate(block_probs):
            block_results[i][other[0][0]] = (
                block_prob.data.cpu().numpy()[0],  # Raw outputs
                F.sigmoid(block_prob).data.cpu().numpy()[0],  # Probabilities
                data[2].numpy()[0],  # Labels
                fps  # Frame rate
            )
    
    return results, block_results

def analyze_block_predictions(model, dataloader, output_dir, epoch):
    """
    Qualitatively analyze what each block is learning.
    
    Args:
        model: The trained model
        dataloader: Validation dataloader
        output_dir: Directory to save analysis results
        epoch: Current epoch number
    """
    model.eval()
    results, block_results = eval_model(model, dataloader)
    
    # Create analysis directories
    analysis_dir = os.path.join(output_dir, 'block_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Analyze each block
    for block_idx, block_data in enumerate(block_results):
        block_analysis = {
            'top_predictions': {},  # Store top predictions per class
            'confusion_matrix': {},  # Store confusion patterns
            'temporal_patterns': {}  # Store temporal activation patterns
        }
        
        # Analyze predictions for each video
        for video_id, (outputs, probs, labels, fps) in block_data.items():
            # Find top predictions
            for class_idx in range(probs.shape[1]):
                if labels[class_idx] > 0:  # For ground truth positive cases
                    if class_idx not in block_analysis['top_predictions']:
                        block_analysis['top_predictions'][class_idx] = []
                    
                    # Store prediction confidence and temporal pattern
                    block_analysis['top_predictions'][class_idx].append({
                        'video_id': video_id,
                        'confidence': float(probs[:, class_idx].max()),
                        'temporal_pattern': probs[:, class_idx].tolist()
                    })
            
            # Analyze confusion patterns
            pred_classes = (probs > 0.5).astype(int)
            for i in range(len(labels)):
                if labels[i] != pred_classes[:, i].any():
                    key = f"{int(labels[i])}_{int(pred_classes[:, i].any())}"
                    if key not in block_analysis['confusion_matrix']:
                        block_analysis['confusion_matrix'][key] = 0
                    block_analysis['confusion_matrix'][key] += 1
            
            # Analyze temporal patterns
            block_analysis['temporal_patterns'][video_id] = {
                'predictions': probs.tolist(),
                'labels': labels.tolist(),
                'fps': float(fps)
            }
        
        # Save block analysis
        block_analysis_file = os.path.join(analysis_dir, f'block_{block_idx+1}_analysis_epoch_{epoch}.pkl')
        pickle.dump(block_analysis, open(block_analysis_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        
        # Log summary statistics
        logging.info(f"\nBlock {block_idx+1} Analysis Summary (Epoch {epoch}):")
        logging.info("-" * 50)
        
        # Log top performing classes
        top_classes = sorted(
            block_analysis['top_predictions'].items(),
            key=lambda x: np.mean([p['confidence'] for p in x[1]]),
            reverse=True
        )[:5]
        
        logging.info("Top 5 Best Predicted Classes:")
        for class_idx, predictions in top_classes:
            avg_conf = np.mean([p['confidence'] for p in predictions])
            logging.info(f"Class {class_idx}: Avg Confidence = {avg_conf:.4f}")
        
        # Log most common confusion patterns
        logging.info("\nTop Confusion Patterns:")
        top_confusions = sorted(
            block_analysis['confusion_matrix'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for pattern, count in top_confusions:
            true_label, pred_label = pattern.split('_')
            logging.info(f"True {true_label} predicted as {pred_label}: {count} times")
        
        logging.info("-" * 50)


def run_network(model, data, gpu, epoch=0, baseline=False):
    # 
    inputs, mask, labels, other, hm = data
    # wrap them in Variable 
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))
    hm = Variable(hm.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)

    outputs_final, block_outputs, diversity_loss = model(inputs)
    
    # Logit for final output
    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)
    
    # Compute loss for final output
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
    loss_f = torch.sum(loss_f) / torch.sum(mask)
    
    # Compute loss for each block
    block_losses = []
    block_probs = []
    for block_output in block_outputs:
        block_prob = F.sigmoid(block_output) * mask.unsqueeze(2)
        block_loss = F.binary_cross_entropy_with_logits(block_output, labels, size_average=False)
        block_loss = torch.sum(block_loss) / torch.sum(mask)
        block_losses.append(block_loss)
        block_probs.append(block_prob)

    # Combine all losses including diversity loss
    block_loss_weight = 0.3  # Weight for each block's loss
    diversity_loss_weight = 100.0 #100.0  # Weight for diversity loss 100.0
    total_block_loss = sum(block_losses) * block_loss_weight
    loss = args.alpha_l * (loss_f + total_block_loss) + diversity_loss_weight * diversity_loss
    
    corr = torch.sum(mask)
    tot = torch.sum(mask)
    
    return outputs_final, loss, probs_f, corr / tot, block_probs, block_losses


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    tot_diversity_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    block_apms = [APMeter() for _ in range(3)]  # One for each block
    
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1
        outputs, loss, probs, err, block_probs, block_losses = run_network(model, data, gpu, epoch)
        
        # Extract diversity loss from the model output for logging
        with torch.no_grad():
            inputs, mask, labels, other, hm = data
            inputs = inputs.squeeze(3).squeeze(3).cuda(gpu)
            _, _, diversity_loss = model(inputs)
            tot_diversity_loss += diversity_loss.item()
        
        # Add metrics for final output
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        
        # Add metrics for each block
        for i, block_prob in enumerate(block_probs):
            block_apms[i].add(block_prob.data.cpu().numpy()[0], data[2].numpy()[0])
        
        error += err.data
        tot_loss += loss.data

        loss.backward()
        optimizer.step()

    # Calculate mAP for final output
    train_map = 100 * apm.value().mean()
    logging.info(f'Epoch {epoch}, train-map: {train_map:.4f}')
    
    # Calculate mAP for each block
    block_maps = []
    for i, block_apm in enumerate(block_apms):
        block_map = 100 * block_apm.value().mean()
        block_maps.append(block_map)
        logging.info(f'Epoch {epoch}, Block {i+1} train-map: {block_map:.4f}')
        block_apm.reset()
    
    apm.reset()
    epoch_loss = tot_loss / num_iter
    avg_diversity_loss = tot_diversity_loss / num_iter
    
    # Log diversity loss
    logging.info(f'Epoch {epoch}, Diversity Loss: {avg_diversity_loss:.6f}')

    return train_map, epoch_loss, block_maps, avg_diversity_loss


def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    sampled_apm = APMeter()
    block_apms = [APMeter() for _ in range(3)]  # One for each block
    block_sampled_apms = [APMeter() for _ in range(3)]  # One for each block
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    full_probs = {}
    block_full_probs = [{} for _ in range(3)]  # One dict for each block

    # Create output directories for each block
    for i in range(3):
        block_dir = os.path.join(args.output_dir, f'block_{i+1}')
        os.makedirs(block_dir, exist_ok=True)

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err, block_probs, block_losses = run_network(model, data, gpu, epoch)
        
        # Process final output
        if sum(data[1].numpy()[0])>25:
            p1,l1=sampled_25(probs.data.cpu().numpy()[0],data[2].numpy()[0],data[1].numpy()[0])
            sampled_apm.add(p1,l1)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        
        # Process block outputs
        for i, block_prob in enumerate(block_probs):
            block_apms[i].add(block_prob.data.cpu().numpy()[0], data[2].numpy()[0])
            if sum(data[1].numpy()[0])>25:
                p1,l1=sampled_25(block_prob.data.cpu().numpy()[0],data[2].numpy()[0],data[1].numpy()[0])
                block_sampled_apms[i].add(p1,l1)

        error += err.data
        tot_loss += loss.data
        
        # Save probabilities for final output and blocks
        probs_1 = mask_probs(probs.data.cpu().numpy()[0],data[1].numpy()[0]).squeeze()
        full_probs[other[0][0]] = probs_1.T
        
        for i, block_prob in enumerate(block_probs):
            block_prob_masked = mask_probs(block_prob.data.cpu().numpy()[0],data[1].numpy()[0]).squeeze()
            block_full_probs[i][other[0][0]] = block_prob_masked.T

    epoch_loss = tot_loss / num_iter
    
    # Calculate metrics for final output
    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    sample_val_map = torch.sum(100 * sampled_apm.value()) / torch.nonzero(100 * sampled_apm.value()).size()[0]

    # Calculate metrics for each block
    block_val_maps = []
    block_sample_val_maps = []
    for i in range(3):
        block_val_map = torch.sum(100 * block_apms[i].value()) / torch.nonzero(100 * block_apms[i].value()).size()[0]
        block_sample_val_map = torch.sum(100 * block_sampled_apms[i].value()) / torch.nonzero(100 * block_sampled_apms[i].value()).size()[0]
        block_val_maps.append(block_val_map)
        block_sample_val_maps.append(block_sample_val_map)
        
        # Log block metrics
        logging.info(f'Epoch {epoch}, Block {i+1} Full-val-map: {block_val_map:.4f}')
        logging.info(f'Epoch {epoch}, Block {i+1} sampled-val-map: {block_sample_val_map:.4f}')
        logging.info(f'Block {i+1} Sampled AP values: {100 * block_sampled_apms[i].value()}')
        
        # Save block probabilities
        block_dir = os.path.join(args.output_dir, f'block_{i+1}')
        pickle.dump(block_full_probs[i], open(os.path.join(block_dir, f'{epoch}.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
        
        block_apms[i].reset()
        block_sampled_apms[i].reset()

    logging.info(f'Epoch {epoch}, Final Full-val-map: {val_map:.4f}')
    logging.info(f'Epoch {epoch}, Final sampled-val-map: {sample_val_map:.4f}')
    logging.info(f'Final Sampled AP values: {100 * sampled_apm.value()}')
    
    apm.reset()
    sampled_apm.reset()
    
    return full_probs, epoch_loss, val_map, sample_val_map, block_val_maps, block_sample_val_maps


def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


if __name__ == '__main__':
    if str(args.unisize) == "True":
        print("uni-size padd all T to",args.num_clips)
        from charades_dataloader import collate_fn_unisize
        collate_fn_f = collate_fn_unisize(args.num_clips)
        collate_fn = collate_fn_f.charades_collate_fn_unisize
    else:
        from charades_dataloader import mt_collate_fn as collate_fn
    
    if args.dataset == 'charades':
        train_split = '/data/asinha13/projects/MAD/MS-TCT/data/charades.json'
        test_split = train_split
        rgb_root =  args.rgb_root 
        flow_root = '/flow_feat_path/' # optional
        classes = 157
        
    elif args.dataset == 'tsu':
        train_split = '/data/asinha13/projects/MAD/MS-TCT/data/smarthome_CS_51.json'
        test_split = train_split
        rgb_root =  args.rgb_root 
        flow_root = '/flow_feat_path/' # optional
        classes = 51

    if args.mode == 'flow':
        print('flow mode', flow_root)
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'rgb':
        print('RGB mode', rgb_root)
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    setup_logging(args.output_dir)
    logging.info(f"Arguments: {args}")

    if args.train:
        if args.backbone == 'i3d':
            in_feat_dim = 1024
        elif args.backbone == 'clip':
            in_feat_dim = 768

        # Create model using timm's create_model function
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            in_feat_dim=in_feat_dim
        )
        model.cuda()

        criterion = LabelSmoothingCrossEntropy()
 
        optimizer = create_optimizer(args, model)
        lr_scheduler, _ = create_scheduler(args, optimizer)

        if args.model_ema:
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume=''
            )
        else:
            model_ema = None

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Number of parameters: {n_parameters}")

        run([(model, 0, dataloaders, optimizer, lr_scheduler, args.comp_info)], criterion, num_epochs=int(args.epochs))
