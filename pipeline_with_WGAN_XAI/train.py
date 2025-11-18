#!/usr/bin/env python
"""
Simple Command-Line Interface for Training Models

Usage:
    python train.py cnn_bilstm                    # Train with defaults
    python train.py cnn_bilstm --epochs 100       # Custom epochs
    python train.py afib_reslstm --lr 0.0005      # Custom learning rate
    python train.py --list                        # List all models
    python train.py --compare                     # Train all models
"""

import argparse
from universal_trainer import train_model, ModelRegistry
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(
        description='Universal AFib Model Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python train.py --list
  
  # Train CNN-BiLSTM with default settings
  python train.py cnn_bilstm
  
  # Train with custom hyperparameters
  python train.py cnn_bilstm --epochs 100 --lr 0.001 --batch_size 64
  
  # Train on CPU
  python train.py cnn_bilstm --device cpu
  
  # Quick test (5 epochs)
  python train.py cnn_bilstm --epochs 5 --quick
  
  # Compare all models
  python train.py --compare --epochs 50
        """
    )
    
    parser.add_argument('model', type=str, nargs='?', help='Model name to train')
    parser.add_argument('--list', action='store_true', help='List all available models')
    parser.add_argument('--compare', action='store_true', help='Train all models for comparison')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda/cpu/auto (default: auto)')
    parser.add_argument('--data_path', type=str, default='../data/processed/', help='Path to processed data')
    parser.add_argument('--save_dir', type=str, default='../results/', help='Results save directory')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (overrides epochs to 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.epochs = 5
        print("⚡ Quick test mode: Training for 5 epochs only\n")
    
    # List models
    if args.list:
        print("\n" + "="*70)
        print("🔍 AVAILABLE MODELS")
        print("="*70)
        registry = ModelRegistry()
        registry.list_models()
        print("\n" + "="*70)
        print("💡 Usage: python train.py <model_name> [options]")
        print("="*70 + "\n")
        return
    
    # Compare all models
    if args.compare:
        print("\n" + "="*70)
        print("📊 COMPARING ALL MODELS")
        print("="*70 + "\n")
        
        registry = ModelRegistry()
        all_models = list(registry.registry.keys())
        
        print(f"Will train {len(all_models)} models:")
        for model_name in all_models:
            print(f"  • {model_name}")
        
        print(f"\nConfiguration:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning Rate: {args.lr}")
        print(f"  Batch Size: {args.batch_size}")
        
        input("\nPress Enter to start, or Ctrl+C to cancel...")
        
        comparison_results = {}
        
        for i, model_name in enumerate(all_models, 1):
            print(f"\n{'='*70}")
            print(f"MODEL {i}/{len(all_models)}: {model_name}")
            print(f"{'='*70}\n")
            
            try:
                results = train_model(
                    model_name=model_name,
                    data_path=args.data_path,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    device=args.device,
                    save_dir=args.save_dir,
                    early_stopping_patience=args.patience,
                    random_seed=args.seed
                )
                
                comparison_results[model_name] = {
                    'auroc': results['test_metrics']['auroc'],
                    'f1_score': results['test_metrics']['f1_score'],
                    'sensitivity': results['test_metrics']['sensitivity'],
                    'specificity': results['test_metrics']['specificity'],
                    'accuracy': results['test_metrics']['accuracy']
                }
                
                print(f"\n✅ {model_name} complete!")
                
            except Exception as e:
                print(f"\n❌ {model_name} failed: {str(e)}")
                comparison_results[model_name] = {'error': str(e)}
        
        # Save comparison results
        comparison_path = Path(args.save_dir) / 'model_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("📊 COMPARISON SUMMARY")
        print("="*70 + "\n")
        
        print(f"{'Model':<30} {'AUROC':>8} {'F1':>8} {'Sens':>8} {'Spec':>8}")
        print("-"*70)
        
        for model_name, metrics in comparison_results.items():
            if 'error' not in metrics:
                print(f"{model_name:<30} {metrics['auroc']:>8.4f} {metrics['f1_score']:>8.4f} "
                      f"{metrics['sensitivity']:>8.4f} {metrics['specificity']:>8.4f}")
        
        # Find best model
        valid_models = {k: v for k, v in comparison_results.items() if 'error' not in v}
        if valid_models:
            best_model = max(valid_models.items(), key=lambda x: x[1]['auroc'])
            print("\n" + "="*70)
            print(f"🏆 BEST MODEL: {best_model[0]}")
            print(f"   AUROC: {best_model[1]['auroc']:.4f}")
            print(f"   F1: {best_model[1]['f1_score']:.4f}")
            print("="*70)
        
        print(f"\n💾 Comparison saved to: {comparison_path}\n")
        return
    
    # Single model training
    if not args.model:
        parser.print_help()
        return
    
    print("\n" + "="*70)
    print(f"🚀 TRAINING: {args.model}")
    print("="*70 + "\n")
    
    results = train_model(
        model_name=args.model,
        data_path=args.data_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        save_dir=args.save_dir,
        early_stopping_patience=args.patience,
        random_seed=args.seed
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Test Results:")
    print(f"   AUROC:       {results['test_metrics']['auroc']:.4f}")
    print(f"   F1-Score:    {results['test_metrics']['f1_score']:.4f}")
    print(f"   Sensitivity: {results['test_metrics']['sensitivity']:.4f}")
    print(f"   Specificity: {results['test_metrics']['specificity']:.4f}")
    print(f"   Accuracy:    {results['test_metrics']['accuracy']:.4f}")
    
    save_path = Path(args.save_dir) / args.model
    print(f"\n💾 Results saved to: {save_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()