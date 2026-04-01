# DOES NOT WORK RIGHT, FIRST VERSION! (even don't fix encryption stuff!)

import struct
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
import argparse
import sys

warnings.filterwarnings('ignore')

@dataclass
class DNNILayer:
    name: str
    layer_type: str
    weights: np.ndarray
    biases: Optional[np.ndarray] = None
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    offset: int = 0
    weight_stats: Dict = field(default_factory=dict)

@dataclass
class DNNIModel:
    model_name: str
    layers: List[DNNILayer]
    metadata: Dict
    version: str
    file_size: int = 0

class DNNIParser:
    FLOAT32_SIZE = 4
    FLOAT64_SIZE = 8
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data = None
        self.layers = []
        self.metadata = {}
        
    def load_binary(self) -> bytes:
        with open(self.file_path, 'rb') as f:
            self.data = f.read()
        return self.data
    
    def detect_float_section(self, offset: int, length: int) -> np.ndarray:
        section_data = self.data[offset:offset + length]
        
        try:
            num_floats = len(section_data) // self.FLOAT32_SIZE
            weights = np.frombuffer(section_data[:num_floats * self.FLOAT32_SIZE], 
                                   dtype=np.float32)
            if np.all(np.isfinite(weights)) and len(weights) > 0:
                return weights
        except:
            pass
        
        return np.array([])
    
    def find_weight_matrices(self, min_size: int = 32) -> List[Dict]:
        matrices = []
        i = 0
        
        while i < len(self.data) - self.FLOAT32_SIZE:
            try:
                float_val = struct.unpack('<f', self.data[i:i+4])[0]
                
                if np.isfinite(float_val) and abs(float_val) < 100:
                    start = i
                    count = 0
                    
                    while i < len(self.data) - 4:
                        val = struct.unpack('<f', self.data[i:i+4])[0]
                        if np.isfinite(val) and abs(val) < 100:
                            count += 1
                            i += 4
                        else:
                            break
                    
                    if count >= min_size:
                        weights = self.detect_float_section(start, count * 4)
                        
                        if len(weights) == count and len(weights) > 0:
                            matrices.append({
                                'offset': start,
                                'size': count,
                                'weights': weights,
                                'shape': self._infer_shape(count),
                                'stats': self._compute_stats(weights)
                            })
            except:
                i += 1
            i += 1
        
        return matrices
    
    def _infer_shape(self, size: int) -> Tuple[int, int]:
        common_dims = [8, 16, 32, 64, 128, 256, 512, 768, 1024]
        
        for out_dim in common_dims:
            if size % out_dim == 0:
                in_dim = size // out_dim
                if in_dim in common_dims or (in_dim >= 2 and in_dim <= 512):
                    return (out_dim, in_dim)
        
        f0_input_dims = [15, 16, 17, 32, 64]
        for in_dim in f0_input_dims:
            if size % in_dim == 0:
                out_dim = size // in_dim
                if out_dim >= 8 and out_dim <= 512:
                    return (out_dim, in_dim)
        
        sqrt_size = int(np.sqrt(size))
        for dim in range(sqrt_size, 0, -1):
            if size % dim == 0:
                return (size // dim, dim)
        
        return (1, size)
    
    def _compute_stats(self, weights: np.ndarray) -> Dict:
        return {
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'median': float(np.median(weights)),
            'q25': float(np.percentile(weights, 25)),
            'q75': float(np.percentile(weights, 75)),
            'sparsity': float(np.mean(np.abs(weights) < 1e-6)),
            'non_zero': int(np.count_nonzero(weights))
        }
    
    def parse_header(self) -> Dict:
        header = {}
        if len(self.data) >= 16:
            header['file_size'] = len(self.data)
            header['first_bytes'] = self.data[:16].hex()
            header['magic'] = self.data[:8]
        try:
            text_sample = self.data[:256].decode('ascii', errors='ignore')
            if 'DNNI' in text_sample or 'Dreamtonics' in text_sample:
                header['format'] = 'DNNI'
        except:
            pass
        return header
    
    def extract_model(self) -> DNNIModel:
        self.load_binary()
        self.metadata = self.parse_header()
        matrices = self.find_weight_matrices()
        
        for i, matrix in enumerate(matrices):
            expected_size = matrix['shape'][0] * matrix['shape'][1]
            actual_size = len(matrix['weights'])
            
            if expected_size != actual_size:
                matrix['shape'] = self._infer_shape(actual_size)
            
            try:
                weights_reshaped = matrix['weights'].reshape(matrix['shape'])
            except ValueError as e:
                print(f"⚠ Warning: Layer {i} reshape failed, using 1D: {e}")
                matrix['shape'] = (1, actual_size)
                weights_reshaped = matrix['weights'].reshape(matrix['shape'])
            
            layer = DNNILayer(
                name=f"layer_{i}",
                layer_type="dense",
                weights=weights_reshaped,
                input_shape=matrix['shape'][1],
                output_shape=matrix['shape'][0],
                offset=matrix['offset'],
                weight_stats=matrix['stats']
            )
            self.layers.append(layer)
        
        return DNNIModel(
            model_name=self.file_path.stem,
            layers=self.layers,
            metadata=self.metadata,
            version="1.0",
            file_size=len(self.data)
        )


class DNNIVisualizer:
    def __init__(self, model: DNNIModel, output_dir: str = "./viz"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_weight_distribution(self, layer_indices: Optional[List[int]] = None, 
                                save: bool = True) -> Optional[plt.Figure]:
        if layer_indices is None:
            layer_indices = list(range(min(16, len(self.model.layers))))
        
        if not layer_indices:
            print("⚠ No layers to visualize")
            return None
        
        n_plots = len(layer_indices)
        cols = 4
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5*rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, ax in zip(layer_indices, axes):
            if idx >= len(self.model.layers):
                ax.axis('off')
                continue
            
            layer = self.model.layers[idx]
            weights = layer.weights.flatten()
            
            ax.hist(weights, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(layer.weight_stats['mean'], color='red', linestyle='--', 
                      label=f"Mean: {layer.weight_stats['mean']:.3f}")
            ax.axvline(layer.weight_stats['q25'], color='orange', linestyle=':', alpha=0.5)
            ax.axvline(layer.weight_stats['q75'], color='orange', linestyle=':', 
                      alpha=0.5, label=f"Q25-Q75")
            ax.set_title(f"{layer.name}\nShape: {layer.weights.shape}", fontsize=9)
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Weight Distributions - {self.model.model_name}", 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "weight_distributions.png", 
                       dpi=150, bbox_inches='tight')
            print(f"✓ Saved weight distributions to {self.output_dir / 'weight_distributions.png'}")
        
        return fig
    
    def plot_layer_statistics(self, save: bool = True) -> Optional[plt.Figure]:
        if not self.model.layers:
            print("⚠ No layers to visualize")
            return None
        
        stats = [l.weight_stats for l in self.model.layers]
        x_pos = np.arange(len(stats))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        mins = [s['min'] for s in stats]
        maxs = [s['max'] for s in stats]
        axes[0, 0].fill_between(x_pos, mins, maxs, alpha=0.3, color='skyblue')
        axes[0, 0].plot(x_pos, mins, 'b-', label='Min', linewidth=1)
        axes[0, 0].plot(x_pos, maxs, 'r-', label='Max', linewidth=1)
        axes[0, 0].set_title("Weight Range per Layer")
        axes[0, 0].set_xlabel("Layer Index")
        axes[0, 0].set_ylabel("Weight Value")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        means = [s['mean'] for s in stats]
        stds = [s['std'] for s in stats]
        axes[0, 1].plot(x_pos, means, color='green', marker='o', label='Mean', 
                       markersize=3, linestyle='-')
        axes[0, 1].fill_between(x_pos, 
                               np.array(means) - np.array(stds),
                               np.array(means) + np.array(stds),
                               alpha=0.3, color='lightgreen', label='±1 Std')
        axes[0, 1].set_title("Mean ± Std Deviation")
        axes[0, 1].set_xlabel("Layer Index")
        axes[0, 1].set_ylabel("Value")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        sparsity = [s['sparsity'] for s in stats]
        axes[0, 2].bar(x_pos, sparsity, color='coral', edgecolor='black')
        axes[0, 2].set_title("Weight Sparsity")
        axes[0, 2].set_xlabel("Layer Index")
        axes[0, 2].set_ylabel("Sparsity Ratio")
        axes[0, 2].grid(alpha=0.3, axis='y')
        
        sizes = [l.weights.size for l in self.model.layers]
        axes[1, 0].plot(x_pos, sizes, color='purple', marker='o', markersize=3, linestyle='-')
        axes[1, 0].set_title("Layer Parameter Count")
        axes[1, 0].set_xlabel("Layer Index")
        axes[1, 0].set_ylabel("Parameters")
        axes[1, 0].grid(alpha=0.3)
        
        non_zero = [s['non_zero'] for s in stats]
        axes[1, 1].plot(x_pos, non_zero, color='blue', marker='o', markersize=3, linestyle='-')
        axes[1, 1].set_title("Non-zero Weights")
        axes[1, 1].set_xlabel("Layer Index")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(alpha=0.3)
        
        axes[1, 2].axis('off')
        overview_text = f"""
        Model: {self.model.model_name}
        File Size: {self.model.file_size:,} bytes
        Total Layers: {len(self.model.layers)}
        
        Total Parameters: {sum(l.weights.size for l in self.model.layers):,}
        
        Avg Weight Range: [{np.mean([s['min'] for s in stats]):.3f}, 
                           {np.mean([s['max'] for s in stats]):.3f}]
        Avg Sparsity: {np.mean([s['sparsity'] for s in stats])*100:.2f}%
        
        First Layer: {self.model.layers[0].weights.shape if self.model.layers else 'N/A'}
        Last Layer: {self.model.layers[-1].weights.shape if self.model.layers else 'N/A'}
        """
        axes[1, 2].text(0.05, 0.95, overview_text.strip(), 
                       fontsize=9, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                       family='monospace')
        
        plt.suptitle(f"Layer Statistics Overview - {self.model.model_name}", 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "layer_statistics.png", 
                       dpi=150, bbox_inches='tight')
            print(f"✓ Saved layer statistics to {self.output_dir / 'layer_statistics.png'}")
        
        return fig
    
    def plot_weight_heatmaps(self, layer_indices: Optional[List[int]] = None,
                           max_layers: int = 8, save: bool = True) -> Optional[plt.Figure]:
        if layer_indices is None:
            layer_indices = list(range(min(max_layers, len(self.model.layers))))
        
        if not layer_indices:
            print("⚠ No layers to visualize")
            return None
        
        n_plots = len(layer_indices)
        cols = 4
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, ax in zip(layer_indices, axes):
            if idx >= len(self.model.layers):
                ax.axis('off')
                continue
            
            layer = self.model.layers[idx]
            weights = layer.weights
            
            if weights.size > 0:
                w_norm = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-8)
                im = ax.imshow(w_norm, cmap='RdBu_r', aspect='auto', interpolation='nearest')
                ax.set_title(f"{layer.name}\n{weights.shape}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, "Empty", ha='center', va='center')
                ax.set_title(layer.name)
        
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Weight Matrix Heatmaps - {self.model.model_name}", 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "weight_heatmaps.png", 
                       dpi=150, bbox_inches='tight')
            print(f"✓ Saved weight heatmaps to {self.output_dir / 'weight_heatmaps.png'}")
        
        return fig
    
    def plot_architecture_diagram(self, save: bool = True) -> Optional[plt.Figure]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.axis('off')
        
        if not self.model.layers:
            ax.text(0.5, 0.5, "No layers to display", ha='center', va='center', fontsize=14)
            return fig
        
        y_pos = 0.95
        layer_num = 0
        
        for i, layer in enumerate(self.model.layers[:15]):
            ax.text(0.1, y_pos, f"Input", ha='right', va='center', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), fontsize=10)
            
            ax.text(0.35, y_pos - 0.03, f"Layer {i}", 
                   ha='center', va='top', fontsize=9, fontweight='bold')
            ax.text(0.35, y_pos - 0.07, f"{layer.weights.shape}", 
                   ha='center', va='top', fontsize=8, style='italic')
            
            rect = plt.Rectangle((0.25, y_pos-0.12), 0.2, 0.1, 
                                facecolor='lightgreen', edgecolor='darkgreen', 
                                alpha=0.6, linewidth=1.5)
            ax.add_patch(rect)
            
            ax.plot([0.2, 0.25], [y_pos, y_pos-0.07], 'k--', alpha=0.5)
            ax.plot([0.45, 0.5], [y_pos-0.07, y_pos-0.07], 'k--', alpha=0.5)
            ax.annotate('', xy=(0.52, y_pos-0.07), xytext=(0.45, y_pos-0.07),
                       arrowprops=dict(arrowstyle='->', color='gray'))
            
            ax.text(0.65, y_pos-0.07, f"Output", 
                   ha='left', va='center', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
            
            y_pos -= 0.15
            layer_num += 1
        
        info_text = f"""
        Architecture Summary:
        • Total Layers: {len(self.model.layers)}
        • Total Parameters: {sum(l.weights.size for l in self.model.layers):,}
        • File: {self.model.model_name}
        """
        ax.text(0.02, 0.02, info_text, fontsize=8, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               family='monospace')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Model Architecture - {self.model.model_name}", 
                    fontsize=14, fontweight='bold', pad=20)
        
        if save:
            plt.savefig(self.output_dir / "architecture_diagram.png", 
                       dpi=150, bbox_inches='tight')
            print(f"✓ Saved architecture diagram to {self.output_dir / 'architecture_diagram.png'}")
        
        return fig
    
    def plot_all(self) -> Dict[str, str]:
        print(f"\n🎨 Generating visualizations for {self.model.model_name}...")
        
        saved_files = {}
        
        try:
            self.plot_weight_distribution(save=True)
            saved_files['distributions'] = str(self.output_dir / "weight_distributions.png")
        except Exception as e:
            print(f"⚠ Error plotting distributions: {e}")
        
        try:
            self.plot_layer_statistics(save=True)
            saved_files['statistics'] = str(self.output_dir / "layer_statistics.png")
        except Exception as e:
            print(f"⚠ Error plotting statistics: {e}")
        
        try:
            self.plot_weight_heatmaps(save=True)
            saved_files['heatmaps'] = str(self.output_dir / "weight_heatmaps.png")
        except Exception as e:
            print(f"⚠ Error plotting heatmaps: {e}")
        
        try:
            self.plot_architecture_diagram(save=True)
            saved_files['architecture'] = str(self.output_dir / "architecture_diagram.png")
        except Exception as e:
            print(f"⚠ Error plotting architecture: {e}")
        
        print(f"\n✓ Visualizations saved to: {self.output_dir}")
        return saved_files
    
    def interactive_summary(self):
        print(f"\n📊 Model Summary: {self.model.model_name}")
        print(f"   File Size: {self.model.file_size:,} bytes")
        print(f"   Layers: {len(self.model.layers)}")
        print(f"   Total Parameters: {sum(l.weights.size for l in self.model.layers):,}")
        
        if self.model.layers:
            print(f"\n   Layer Details:")
            for i, layer in enumerate(self.model.layers[:10]):
                stats = layer.weight_stats
                print(f"   [{i:2d}] {layer.name:12s} {str(layer.weights.shape):12s} | "
                      f"μ={stats['mean']:6.3f} σ={stats['std']:6.3f} | "
                      f"range=[{stats['min']:6.3f}, {stats['max']:6.3f}] | "
                      f"sparsity={stats['sparsity']*100:5.1f}%")
            if len(self.model.layers) > 10:
                print(f"   ... and {len(self.model.layers) - 10} more layers")


def analyze_dnni_structure(file_path: str):
    parser = DNNIParser(file_path)
    parser.load_binary()
    
    print(f"File size: {len(parser.data)} bytes")
    print(f"Header info: {parser.parse_header()}")
    
    matrices = parser.find_weight_matrices(min_size=32)
    print(f"\nFound {len(matrices)} potential weight matrices:")
    
    for i, m in enumerate(matrices[:10]):
        print(f"  Matrix {i}: offset={m['offset']}, size={m['size']}, shape={m['shape']}")
        print(f"    Weight range: [{m['weights'].min():.4f}, {m['weights'].max():.4f}]")
        print(f"    Mean: {m['weights'].mean():.4f}, Std: {m['weights'].std():.4f}")
    
    if len(matrices) > 10:
        print(f"  ... and {len(matrices) - 10} more matrices")
    
    return parser


def save_weights(model: DNNIModel, output_dir: str) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    for i, layer in enumerate(model.layers):
        weights_file = output_path / f"{layer.name}_weights.npy"
        np.save(weights_file, layer.weights)
        saved_files[layer.name] = str(weights_file)
        
        if layer.biases is not None:
            biases_file = output_path / f"{layer.name}_biases.npy"
            np.save(biases_file, layer.biases)
            saved_files[f"{layer.name}_biases"] = str(biases_file)
    
    meta_file = output_path / "model_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump({
            'model_name': model.model_name,
            'num_layers': len(model.layers),
            'total_parameters': sum(l.weights.size for l in model.layers),
            'metadata': {k: (v.hex() if isinstance(v, bytes) else v) for k, v in model.metadata.items()},
            'layers': [
                {
                    'name': l.name,
                    'shape': l.weights.shape,
                    'stats': l.weight_stats
                }
                for l in model.layers
            ]
        }, f, indent=2)
    saved_files['metadata'] = str(meta_file)
    
    print(f"✓ Saved {len(model.layers)} layers to {output_path}")
    print(f"  Total files: {len(saved_files)}")
    
    return saved_files


def visualize_dnni(input_file: str, output_dir: str = "./viz", 
                  save_weights_dir: Optional[str] = None,
                  show_plots: bool = False, plot_types: List[str] = None):
    print(f"🔍 Loading DNNI file: {input_file}")
    
    parser = DNNIParser(input_file)
    model = parser.extract_model()
    
    print(f"✓ Extracted {len(model.layers)} layers")
    
    if not model.layers:
        print("⚠ Warning: No layers extracted! Check file format.")
        return model
    
    if save_weights_dir:
        save_weights(model, save_weights_dir)
    
    viz = DNNIVisualizer(model, output_dir)
    viz.interactive_summary()
    
    if plot_types is None:
        viz.plot_all()
    else:
        if 'distributions' in plot_types:
            viz.plot_weight_distribution(save=True)
        if 'statistics' in plot_types:
            viz.plot_layer_statistics(save=True)
        if 'heatmaps' in plot_types:
            viz.plot_weight_heatmaps(save=True)
        if 'architecture' in plot_types:
            viz.plot_architecture_diagram(save=True)
    
    if show_plots:
        print("\n🖼️  Opening plots...")
        plt.show()
    
    return model


def convert_dnni(input_file: str, output_dir: str, format: str = 'numpy',
                visualize: bool = False, viz_dir: str = "./viz"):
    print(f"Loading DNNI file: {input_file}")
    
    parser = DNNIParser(input_file)
    model = parser.extract_model()
    
    print(f"Extracted {len(model.layers)} layers")
    print(f"Model metadata: {model.metadata}")
    
    if format == 'numpy':
        save_weights(model, output_dir)
    
    if visualize:
        viz = DNNIVisualizer(model, viz_dir)
        viz.plot_all()
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DNNI Format Converter with Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert and save weights (RECOMMENDED)
  python3 dnni_converter.py f0.dnni ./converted/f0
  
  # Convert with visualization
  python3 dnni_converter.py f0.dnni ./converted/f0 --visualize
  
  # Visualize only (no weight saving)
  python3 dnni_converter.py f0.dnni --visualize-only
  
  # Analyze only
  python3 dnni_converter.py --analyze f0.dnni
  
  # Batch convert all models
  python3 dnni_converter.py --batch f0.dnni audio2score.dnni ornamentation.dnni
        """
    )
    
    parser.add_argument("input_file", nargs="?", help="Input DNNI file")
    parser.add_argument("output_dir", nargs="?", help="Output directory for weights")
    parser.add_argument("--format", choices=["numpy", "onnx", "pytorch"], 
                       default="numpy", help="Conversion format")
    parser.add_argument("--visualize", "-v", action="store_true", 
                       help="Generate visualizations alongside weights")
    parser.add_argument("--visualize-only", action="store_true",
                       help="Only create visualizations, don't save weights")
    parser.add_argument("--viz-dir", default="./viz",
                       help="Directory for visualization output")
    parser.add_argument("--show", "-s", action="store_true",
                       help="Display plots interactively")
    parser.add_argument("--plots", "-p", nargs="+", 
                       choices=["distributions", "statistics", "heatmaps", "architecture"],
                       help="Specific plot types to generate")
    parser.add_argument("--analyze", "-a", action="store_true",
                       help="Analyze file structure only")
    parser.add_argument("--batch", nargs="+", help="Batch process multiple DNNI files")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing
        for dnni_file in args.batch:
            model_name = Path(dnni_file).stem
            output_dir = f"./converted/{model_name}"
            viz_dir = f"./viz/{model_name}"
            
            print(f"\n{'='*60}")
            print(f"Processing: {dnni_file}")
            print(f"{'='*60}\n")
            
            convert_dnni(dnni_file, output_dir, args.format, 
                        visualize=True, viz_dir=viz_dir)
        sys.exit(0)
    
    if not args.input_file:
        parser.print_help()
        sys.exit(1)
    
    if args.analyze:
        analyze_dnni_structure(args.input_file)
        sys.exit(0)
    
    if args.visualize_only:
        # Visualization only mode
        visualize_dnni(
            args.input_file, 
            output_dir=args.viz_dir,
            show_plots=args.show,
            plot_types=args.plots
        )
    elif args.output_dir:
        # Conversion mode (with optional visualization)
        convert_dnni(
            args.input_file, 
            args.output_dir, 
            args.format,
            visualize=args.visualize,
            viz_dir=args.viz_dir
        )
    else:
        parser.print_help()
        sys.exit(1)
