import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_all_bleu_scores(args):
    """Calculate and visualize BLEU scores for all models."""
    print("Calculating BLEU scores...")
    
    # Read results files
    results_files = [
        ('models', 'bleu_scores.csv'),
        ('apis', 'api_bleu_scores.csv')
    ]
    
    all_results = []
    
    for subdir, filename in results_files:
        filepath = os.path.join(args.results_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            all_results.append(df)
    
    # Combine results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        output_path = os.path.join(args.results_dir, 'combined_bleu_scores.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"Combined results saved to {output_path}")
        
        # Sort by BLEU score
        combined_df = combined_df.sort_values('BLEU Score', ascending=False)
        print("\nBLEU Scores (sorted):")
        print(combined_df.to_string(index=False))
        
        # Create visualization
        if args.visualize:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=combined_df, x='Model', y='BLEU Score', palette='viridis')
            plt.title('BLEU Score Comparison Across Models', fontsize=16, fontweight='bold')
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('BLEU Score', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plot_path = os.path.join(args.results_dir, 'bleu_scores_plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to {plot_path}")
    else:
        print("No results found. Please run evaluation first.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate and compare BLEU scores')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    
    args = parser.parse_args()
    calculate_all_bleu_scores(args)
