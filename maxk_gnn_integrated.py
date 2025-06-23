#!/usr/bin/env python3

import argparse
import dgl
import dgl.nn as dglnn
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from torch.autograd import Function
import math
import time
from collections import OrderedDict, deque

from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset, FlickrDataset, YelpDataset
from utils.config import TrainConfig
import os
import utils.general_utils as general_utils
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from utils.proteins_loader import load_proteins
from utils.models import SAGE, GCN, GIN, GNN_res

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def load_dataset(config):
    """Load dataset based on config"""
    if config.dataset == "reddit":
        dataset = RedditDataset(self_loop=config.selfloop)
        g = dataset[0]
        g = dgl.add_self_loop(g) if config.selfloop else g
        num_classes = dataset.num_classes
        
    elif config.dataset == "flickr":
        dataset = FlickrDataset()
        g = dataset[0]
        g = dgl.add_self_loop(g) if config.selfloop else g
        num_classes = dataset.num_classes
        
    elif config.dataset == "yelp":
        dataset = YelpDataset()
        g = dataset[0]
        g = dgl.add_self_loop(g) if config.selfloop else g
        num_classes = dataset.num_classes
        
    elif config.dataset == "ogbn-arxiv":
        dataset = DglNodePropPredDataset(name="ogbn-arxiv", root=config.data_path)
        g, labels = dataset[0]
        g = dgl.add_self_loop(g) if config.selfloop else g
        g.ndata['labels'] = labels.squeeze()
        
        # Get split
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True
        
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
        
        num_classes = dataset.num_classes
        
    elif config.dataset == "ogbn-products":
        dataset = DglNodePropPredDataset(name="ogbn-products", root=config.data_path)
        g, labels = dataset[0]
        g = dgl.add_self_loop(g) if config.selfloop else g
        g.ndata['labels'] = labels.squeeze()
        
        # Get split
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True
        
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
        
        num_classes = dataset.num_classes
        
    elif config.dataset == "ogbn-proteins":
        g, num_classes = load_proteins(config.data_path)
        g = dgl.add_self_loop(g) if config.selfloop else g
        
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    return g, num_classes

def evaluate(model, g, features, labels, mask, evaluator=None):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        
        if evaluator is not None:
            # OGB evaluation
            y_pred = logits.argmax(dim=-1, keepdim=True)
            y_true = labels.unsqueeze(-1)
            result = evaluator.eval({
                'y_true': y_true.cpu(),
                'y_pred': y_pred.cpu()
            })
            accuracy = result['acc']
        else:
            # Standard accuracy
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            accuracy = correct.item() * 1.0 / len(labels)
        
        return accuracy

def train(g, features, labels, masks, model, config, logger, writer):
    """Training loop with MaxK-GNN integration"""
    train_mask, val_mask, test_mask = masks
    
    # Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.w_lr, weight_decay=config.w_weight_decay)
    loss_fcn = torch.nn.CrossEntropyLoss()
    
    # Setup evaluator for OGB datasets
    evaluator = None
    if config.dataset.startswith('ogbn'):
        evaluator = Evaluator(name=config.dataset)
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    max_patience = 100
    
    print(f"🚀 Starting training with MaxK-GNN integration")
    print(f"   Model: {config.model.upper()}")
    print(f"   Nonlinearity: {config.nonlinear}")
    print(f"   MaxK: {config.maxk}")
    print(f"   Dataset: {config.dataset}")
    print(f"   Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")
    
    # Check if we're using custom kernels
    try:
        import spmm_kernels
        print(f"✅ Using MaxK-GNN custom kernels")
    except ImportError:
        print(f"⚠ Using DGL/PyTorch fallback")
    
    training_start_time = time.time()
    
    for epoch in range(config.epochs):
        model.train()
        
        # Forward pass
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if epoch % 10 == 0:
            train_acc = evaluate(model, g, features, labels, train_mask, evaluator)
            val_acc = evaluate(model, g, features, labels, val_mask, evaluator)
            test_acc = evaluate(model, g, features, labels, test_mask, evaluator)
            
            # Update best scores
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience = 0
            else:
                patience += 1
            
            # Logging
            logger.info(f"Epoch {epoch:04d} | Loss {loss.item():.4f} | "
                       f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f} | "
                       f"Best Val {best_val_acc:.4f} | Best Test {best_test_acc:.4f}")
            
            # TensorBoard logging
            if writer:
                writer.add_scalar('Loss/Train', loss.item(), epoch)
                writer.add_scalar('Accuracy/Train', train_acc, epoch)
                writer.add_scalar('Accuracy/Val', val_acc, epoch)
                writer.add_scalar('Accuracy/Test', test_acc, epoch)
                writer.add_scalar('Accuracy/Best_Val', best_val_acc, epoch)
                writer.add_scalar('Accuracy/Best_Test', best_test_acc, epoch)
            
            # Early stopping
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    total_training_time = time.time() - training_start_time
    
    # Stop timing and get stats for SAGE
    timing_stats = None
    if isinstance(model, SAGE):
        timing_stats = model.stop_training_timing()
        if timing_stats:
            logger.info(f"Training completed in {total_training_time:.2f}s")
            logger.info(f"Aggregation time: {timing_stats['aggregation_time']:.2f}s "
                       f"({timing_stats['percentage']:.1f}% of total)")
    
    # Final evaluation
    final_train_acc = evaluate(model, g, features, labels, train_mask, evaluator)
    final_val_acc = evaluate(model, g, features, labels, val_mask, evaluator)
    final_test_acc = evaluate(model, g, features, labels, test_mask, evaluator)
    
    results = {
        'train_acc': final_train_acc,
        'val_acc': final_val_acc,
        'test_acc': final_test_acc,
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'total_time': total_training_time,
        'timing_stats': timing_stats
    }
    
    return results

def main():
    """Main training function"""
    config = TrainConfig()
    args, _ = args, unknown = config.parse_known_args()
    
    # Setup logging
    os.makedirs(args.path, exist_ok=True)
    logger = general_utils.get_logger(f"{args.path}/train.log")
    writer = SummaryWriter(args.path) if args.path else None
    
    # Set random seeds
    torch.manual_seed(args.seed)        
    torch.cuda.manual_seed_all(args.seed)  
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    g, num_classes = load_dataset(args)
    
    # Move to device
    g = g.to(device)
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)
    
    # Get masks
    train_mask = g.ndata['train_mask'].to(device)
    val_mask = g.ndata['val_mask'].to(device)
    test_mask = g.ndata['test_mask'].to(device)
    masks = (train_mask, val_mask, test_mask)
    
    # Create model
    in_size = features.shape[1]
    logger.info(f"Creating {args.model.upper()} model")
    logger.info(f"  Input size: {in_size}")
    logger.info(f"  Hidden size: {args.hidden_dim}")
    logger.info(f"  Hidden layers: {args.hidden_layers}")
    logger.info(f"  Output size: {num_classes}")
    logger.info(f"  MaxK: {args.maxk}")
    logger.info(f"  Nonlinearity: {args.nonlinear}")
    
    if args.model == 'sage':
        model = SAGE(in_size, args.hidden_dim, args.hidden_layers, num_classes,
                    maxk=args.maxk, feat_drop=args.dropout, norm=args.norm,
                    nonlinear=args.nonlinear)
    elif args.model == 'gcn':
        model = GCN(in_size, args.hidden_dim, args.hidden_layers, num_classes,
                   maxk=args.maxk, feat_drop=args.dropout, norm=args.norm,
                   nonlinear=args.nonlinear)
    elif args.model == 'gin':
        model = GIN(in_size, args.hidden_dim, args.hidden_layers, num_classes,
                   maxk=args.maxk, feat_drop=args.dropout, norm=args.norm,
                   nonlinear=args.nonlinear)
    elif args.model == 'gnn_res':
        model = GNN_res(in_size, args.hidden_dim, args.hidden_layers, num_classes,
                       maxk=args.maxk, feat_drop=args.dropout, norm=args.norm,
                       nonlinear=args.nonlinear)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Log configuration
    config.print_params(logger.info)
    
    # Train model
    results = train(g, features, labels, masks, model, args, logger, writer)
    
    # Log final results
    logger.info("="*50)
    logger.info("FINAL RESULTS")
    logger.info("="*50)
    logger.info(f"Train Accuracy: {results['train_acc']:.4f}")
    logger.info(f"Val Accuracy: {results['val_acc']:.4f}")
    logger.info(f"Test Accuracy: {results['test_acc']:.4f}")
    logger.info(f"Best Val Accuracy: {results['best_val_acc']:.4f}")
    logger.info(f"Best Test Accuracy: {results['best_test_acc']:.4f}")
    logger.info(f"Total Training Time: {results['total_time']:.2f}s")
    
    if results['timing_stats']:
        logger.info(f"Aggregation Time: {results['timing_stats']['aggregation_time']:.2f}s")
        logger.info(f"Aggregation Percentage: {results['timing_stats']['percentage']:.1f}%")
    
    # Save results
    torch.save({
        'config': vars(args),
        'results': results,
        'model_state_dict': model.state_dict()
    }, f"{args.path}/final_results.pt")
    
    if writer:
        writer.close()
    
    print("🎉 Training completed successfully!")
    print(f"📊 Best Test Accuracy: {results['best_test_acc']:.4f}")
    print(f"⏱️ Total Time: {results['total_time']:.2f}s")

if __name__ == "__main__":
    main()