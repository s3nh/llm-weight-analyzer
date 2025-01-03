import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_model(model_name):
    """Load model with half precision to save memory"""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True 
        )
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def analyze_weight_distribution(model, model_name):
    """Analyze weight distribution across model layers"""
    stats = defaultdict(dict)
    layer_weights = defaultdict(list)
    
    # Iterate through named parameters
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only analyze trainable parameters
            # Convert weights to numpy for analysis
            weights = param.detach().cpu().numpy().flatten()
            
            # Categorize layers
            if 'layer' in name:
                layer_num = name.split('layer')[1].split('.')[0]
                layer_type = 'attention' if 'attention' in name else 'ffn' if 'mlp' in name else 'other'
                key = f"layer_{layer_num}_{layer_type}"
            else:
                key = 'other_params'
            
            # Calculate statistics
            stats[key]['mean'] = float(np.mean(weights))
            stats[key]['std'] = float(np.std(weights))
            stats[key]['min'] = float(np.min(weights))
            stats[key]['max'] = float(np.max(weights))
            stats[key]['sparsity'] = float(np.sum(np.abs(weights) < 1e-6) / len(weights))
            
            layer_weights[key] = weights

    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Weight Distribution Violin Plot
    plt.subplot(2, 1, 1)
    data = []
    labels = []
    for key in sorted(layer_weights.keys()):
        if 'layer' in key:  # Only plot main layers
            data.append(layer_weights[key])
            labels.append(key)
    
    violin_parts = plt.violinplot(data, points=100, vert=False)
    plt.yticks(range(1, len(labels) + 1), labels)
    plt.xlabel('Weight Values')
    plt.title(f'Weight Distribution Across Layers - {model_name}')

    # Plot 2: Statistics Summary
    plt.subplot(2, 1, 2)
    layer_numbers = []
    means = []
    stds = []
    sparsities = []
    
    for key in sorted(stats.keys()):
        if 'layer' in key:
            layer_numbers.append(key)
            means.append(stats[key]['mean'])
            stds.append(stats[key]['std'])
            sparsities.append(stats[key]['sparsity'] * 100)  # Convert to percentage
    
    x = range(len(layer_numbers))
    plt.plot(x, means, 'b-', label='Mean')
    plt.plot(x, stds, 'r-', label='Std Dev')
    plt.plot(x, sparsities, 'g-', label='Sparsity %')
    plt.xticks(x, layer_numbers, rotation=45)
    plt.legend()
    plt.title('Layer Statistics Summary')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_weight_analysis.png')
    plt.close()
    
    return stats

def main():
    # Models to analyze
    models = [
        "meta-llama/Llama-2-7b-hf",  # Llama 2 7B model
        "Qwen/Qwen-7B"               # Qwen 7B model
    ]
    
    for model_name in models:
        print(f"\nAnalyzing {model_name}...")
        model = load_model(model_name)
        
        if model is not None:
            stats = analyze_weight_distribution(model, model_name.split('/')[-1])
            
            # Print summary statistics
            print(f"\nSummary Statistics for {model_name}:")
            print("=" * 50)
            for layer, layer_stats in stats.items():
                if 'layer' in layer:  # Only print main layers
                    print(f"\n{layer}:")
                    print(f"Mean: {layer_stats['mean']:.6f}")
                    print(f"Std Dev: {layer_stats['std']:.6f}")
                    print(f"Sparsity: {layer_stats['sparsity']*100:.2f}%")
            
            # Clear model from memory
            del model
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
