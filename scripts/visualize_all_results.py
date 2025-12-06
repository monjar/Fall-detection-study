"""
Create comparison visualizations for all fall detection approaches.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results() -> Dict:
    """Load all results from previous experiments."""
    results = {}
    
    # Known results from previous experiments
    results['KTH Anomaly (Baseline)'] = {
        'accuracy': 0.4558,
        'fall_detection_rate': 0.44,
        'false_alarm_rate': 0.55,
        'approach': 'Autoencoder only'
    }
    
    results['Original Hybrid'] = {
        'accuracy': 0.4639,
        'fall_detection_rate': 0.48,
        'false_alarm_rate': 0.9491,
        'approach': 'AE + 597 synthetic falls'
    }
    
    results['Improved Hybrid (LSTM)'] = {
        'accuracy': 0.4718,
        'fall_detection_rate': 0.5204,
        'false_alarm_rate': 0.9209,
        'approach': 'AE + BiLSTM + 2,995 synthetic'
    }
    
    results['Fine-Tuned (Reference)'] = {
        'accuracy': 0.9582,
        'fall_detection_rate': 0.96,
        'false_alarm_rate': 0.0342,
        'approach': 'Real Kaggle data'
    }
    
    # Load ensemble results if available
    ensemble_file = Path('results/ensemble_test_results.json')
    if ensemble_file.exists():
        with open(ensemble_file, 'r') as f:
            ensemble_data = json.load(f)
            
        if 'ensemble' in ensemble_data:
            results['Ensemble'] = {
                'accuracy': ensemble_data['ensemble']['accuracy'],
                'fall_detection_rate': ensemble_data['ensemble']['fall_detection_rate'],
                'false_alarm_rate': ensemble_data['ensemble']['false_alarm_rate'],
                'approach': 'Multi-detector fusion'
            }
            
            # Add individual detectors
            for name in ['physics', 'temporal', 'geometry']:
                if name in ensemble_data:
                    results[f'Ensemble-{name.capitalize()}'] = {
                        'accuracy': ensemble_data[name]['accuracy'],
                        'fall_detection_rate': ensemble_data[name]['fall_detection_rate'],
                        'false_alarm_rate': ensemble_data[name]['false_alarm_rate'],
                        'approach': f'{name.capitalize()} detector only'
                    }
    
    return results


def plot_accuracy_comparison(results: Dict, output_dir: Path):
    """Plot accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in names]
    
    # Color code: red for poor (<60%), yellow for medium (60-85%), green for good (>85%)
    colors = []
    for acc in accuracies:
        if acc < 60:
            colors.append('#E74C3C')  # Red
        elif acc < 85:
            colors.append('#F39C12')  # Orange
        else:
            colors.append('#27AE60')  # Green
    
    bars = ax.bar(range(len(names)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Fall Detection Accuracy: All Approaches', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 50% (random chance baseline)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Random Chance (50%)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")
    plt.close()


def plot_false_alarm_rates(results: Dict, output_dir: Path):
    """Plot false alarm rate comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(results.keys())
    far_rates = [results[name]['false_alarm_rate'] * 100 for name in names]
    
    # Color code: green for low (<10%), yellow for medium (10-50%), red for high (>50%)
    colors = []
    for far in far_rates:
        if far < 10:
            colors.append('#27AE60')  # Green
        elif far < 50:
            colors.append('#F39C12')  # Orange
        else:
            colors.append('#E74C3C')  # Red
    
    bars = ax.bar(range(len(names)), far_rates, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, far) in enumerate(zip(bars, far_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{far:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('False Alarm Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('False Alarm Rates: All Approaches (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, max(far_rates) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 10% (typical acceptable threshold)
    ax.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Acceptable Threshold (10%)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'false_alarm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'false_alarm_comparison.png'}")
    plt.close()


def plot_accuracy_vs_far(results: Dict, output_dir: Path):
    """Plot accuracy vs false alarm rate scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in names]
    far_rates = [results[name]['false_alarm_rate'] * 100 for name in names]
    
    # Color by approach type
    color_map = {
        'Autoencoder only': '#3498DB',
        'AE + 597 synthetic falls': '#9B59B6',
        'AE + BiLSTM + 2,995 synthetic': '#E74C3C',
        'Real Kaggle data': '#27AE60',
        'Multi-detector fusion': '#F39C12',
    }
    
    colors = [color_map.get(results[name].get('approach', ''), '#95A5A6') for name in names]
    
    scatter = ax.scatter(far_rates, accuracies, c=colors, s=200, alpha=0.7, edgecolors='black', linewidths=2)
    
    # Add labels
    for i, name in enumerate(names):
        ax.annotate(name, (far_rates[i], accuracies[i]), 
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('False Alarm Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs False Alarm Rate (Top-Right = Best)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add ideal zone (high accuracy, low FAR)
    ax.axhline(y=85, color='green', linestyle='--', alpha=0.3, label='High Accuracy (>85%)')
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.3, label='Low FAR (<10%)')
    
    # Shade the ideal quadrant
    ax.fill_between([0, 10], 85, 100, alpha=0.1, color='green', label='Ideal Zone')
    
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_far.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_vs_far.png'}")
    plt.close()


def plot_improvement_progression(results: Dict, output_dir: Path):
    """Plot the progression of improvements."""
    # Order by development timeline
    timeline = [
        'KTH Anomaly (Baseline)',
        'Original Hybrid',
        'Improved Hybrid (LSTM)',
        'Ensemble',
        'Fine-Tuned (Reference)'
    ]
    
    # Filter to only include available results
    timeline = [name for name in timeline if name in results]
    
    accuracies = [results[name]['accuracy'] * 100 for name in timeline]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(range(len(timeline)), accuracies, marker='o', markersize=12, 
            linewidth=3, color='#3498DB', label='Test Accuracy')
    
    # Add value labels
    for i, (name, acc) in enumerate(zip(timeline, accuracies)):
        ax.annotate(f'{acc:.2f}%', (i, acc), 
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    # Highlight the synthetic vs real gap
    if 'Fine-Tuned (Reference)' in timeline:
        ref_idx = timeline.index('Fine-Tuned (Reference)')
        for i in range(ref_idx):
            ax.plot([i, ref_idx], [accuracies[i], accuracies[ref_idx]], 
                   'r--', alpha=0.3, linewidth=1)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Development Timeline: Progression of Improvements', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(timeline)))
    ax.set_xticklabels(timeline, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at 50% (random chance)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Random Chance (50%)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_timeline.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'improvement_timeline.png'}")
    plt.close()


def main():
    output_dir = Path('results/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_results()
    
    print(f"\nFound {len(results)} models:")
    for name in results.keys():
        print(f"  - {name}")
    
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(results, output_dir)
    plot_false_alarm_rates(results, output_dir)
    plot_accuracy_vs_far(results, output_dir)
    plot_improvement_progression(results, output_dir)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
