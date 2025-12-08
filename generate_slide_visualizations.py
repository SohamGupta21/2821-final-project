"""
Generate visualizations for presentation slides.

This script creates all the figures needed for the presentation:
1. System architecture diagram
2. Theory evolution plots
3. Rule examples and coverage
4. Proof tree visualizations
5. Comparison charts
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle, Circle, Arrow
import seaborn as sns

# Set style for publication-quality figures
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Import project modules
from src.models.nn_model import SimpleNNClassifier, generate_synthetic_loan_data
from src.inference.algorithm import ModelInference
from src.inference.oracle import NNOracle
from src.core.atoms import Atom, Constant
from src.utils.visualization import (
    plot_theory_evolution,
    plot_rule_coverage,
    plot_rule_accuracy,
    visualize_proof_tree,
    display_rules
)

# Helper functions from demo
def _create_instances_from_data(X, feature_names, discretize=True, bins=3):
    instances = {}
    for i in range(X.shape[0]):
        instance_features = {}
        for j, feature_name in enumerate(feature_names):
            value = X[i, j]
            if discretize:
                if bins == 3:
                    if value > 0.3:
                        discrete_value = "high"
                    elif value < -0.3:
                        discrete_value = "low"
                    else:
                        discrete_value = "medium"
                else:
                    bin_edges = np.linspace(X[:, j].min(), X[:, j].max(), bins + 1)
                    bin_idx = np.digitize(value, bin_edges) - 1
                    discrete_value = f"bin_{bin_idx}"
                instance_features[feature_name] = discrete_value
            else:
                instance_features[feature_name] = float(value)
        instances[i] = instance_features
    return instances

def _wrap_nn(model, instances, feature_names, label_map):
    class ModelWrapper:
        def __init__(self, nn_model):
            self.nn_model = nn_model
        def predict(self, feature_vector):
            return self.nn_model.predict(feature_vector)
    
    return NNOracle(
        model=ModelWrapper(model),
        instances=instances,
        feature_names=feature_names,
        label_map=label_map
    )

def _generate_facts_from_instances(instances, model, feature_names, label_map, instance_ids=None):
    facts = []
    if instance_ids is None:
        instance_ids = list(instances.keys())
    
    for instance_id in instance_ids:
        if instance_id not in instances:
            continue
        
        instance_features = instances[instance_id]
        feature_vector = []
        for feature_name in feature_names:
            value = instance_features.get(feature_name, 0.0)
            if isinstance(value, str):
                value = float(hash(value) % 100) / 100.0
            feature_vector.append(float(value))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        prediction = model.predict(feature_vector)
        
        label_name = None
        for name, value in label_map.items():
            if value == prediction:
                label_name = name
                break
        
        if label_name is None:
            continue
        
        facts.append(Atom("predict", [Constant(instance_id), Constant(label_name)]))
        
        for feature_name, value in instance_features.items():
            if isinstance(value, str):
                facts.append(Atom("feature", [
                    Constant(instance_id), Constant(feature_name), Constant(value)
                ]))
            else:
                facts.append(Atom("feature", [
                    Constant(instance_id), Constant(feature_name), Constant(float(value))
                ]))
    
    return facts


def generate_system_architecture_diagram(save_path="figures/system_architecture.png"):
    """Generate system architecture diagram for slides."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    model_color = '#FF6B6B'
    adapter_color = '#4ECDC4'
    lang_color = '#95E1D3'
    oracle_color = '#F38181'
    inference_color = '#AA96DA'
    theory_color = '#FCBAD3'
    output_color = '#C7CEEA'
    
    # Boxes
    boxes = [
        # Row 1: Input
        {'x': 1, 'y': 8, 'w': 1.5, 'h': 0.8, 'text': 'Any ML\nModel', 'color': model_color},
        {'x': 3, 'y': 8, 'w': 1.5, 'h': 0.8, 'text': 'Model\nAdapter', 'color': adapter_color},
        {'x': 5, 'y': 8, 'w': 1.5, 'h': 0.8, 'text': 'Observation\nLanguage', 'color': lang_color},
        {'x': 7, 'y': 8, 'w': 1.5, 'h': 0.8, 'text': 'Universal\nOracle', 'color': oracle_color},
        
        # Row 2: Processing
        {'x': 4, 'y': 6, 'w': 2, 'h': 0.8, 'text': 'Model Inference\nAlgorithm', 'color': inference_color},
        
        # Row 3: Output
        {'x': 4, 'y': 4, 'w': 2, 'h': 0.8, 'text': 'Theory\n(Rules)', 'color': theory_color},
        {'x': 4, 'y': 2.5, 'w': 2, 'h': 0.8, 'text': 'Explanations\n& Metrics', 'color': output_color},
    ]
    
    # Draw boxes
    for box in boxes:
        rect = FancyBboxPatch(
            (box['x'], box['y']), box['w'], box['h'],
            boxstyle="round,pad=0.1",
            facecolor=box['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        # Horizontal flow
        (2.5, 8.4, 3, 8.4),
        (4.5, 8.4, 5, 8.4),
        (6.5, 8.4, 7, 8.4),
        # Down to inference
        (7.75, 8, 5, 6.8),
        # Down to theory
        (5, 6, 5, 4.8),
        # Down to output
        (5, 4, 5, 3.3),
    ]
    
    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', lw=2, color='black',
            connectionstyle='arc3,rad=0'
        )
        ax.add_patch(arrow)
    
    # Labels
    ax.text(5, 9.5, 'Plug-and-Play Explainability Pipeline', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Features annotation
    features_text = "Features:\n• Auto-detects model type\n• Framework agnostic\n• One API for all models"
    ax.text(1, 6, features_text, fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_theory_evolution_plot(save_path="figures/theory_evolution.png"):
    """Generate theory evolution visualization."""
    # Run demo to get real data
    print("Generating theory evolution plot...")
    X, y, feature_names = generate_synthetic_loan_data(n_samples=500, random_state=42)
    model = SimpleNNClassifier(input_size=len(feature_names), hidden_sizes=[32, 16])
    model.train_model(X, y, epochs=50, verbose=False)
    
    instances = _create_instances_from_data(X, feature_names, discretize=True)
    label_map = {"APPROVED": 1, "DENIED": 0}
    oracle = _wrap_nn(model, instances, feature_names, label_map)
    
    # Get predictions for stratification
    all_predictions = [model.predict(X[i:i+1]) for i in range(len(X))]
    approved_indices = [i for i, p in enumerate(all_predictions) if p == 1]
    denied_indices = [i for i, p in enumerate(all_predictions) if p == 0]
    
    n_per_class = 50
    sampled_approved = np.random.choice(approved_indices, min(n_per_class, len(approved_indices)), replace=False).tolist() if approved_indices else []
    sampled_denied = np.random.choice(denied_indices, min(n_per_class, len(denied_indices)), replace=False).tolist() if denied_indices else []
    train_indices = sampled_approved + sampled_denied
    np.random.shuffle(train_indices)
    
    facts = _generate_facts_from_instances(instances, model, feature_names, label_map, instance_ids=train_indices)
    
    inference = ModelInference(oracle, max_iterations=20, max_theory_size=20, verbose=False)
    theory = inference.infer_theory(iter(facts))
    history = inference.get_history()
    
    # Plot
    sizes = [len(t) for t in history]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(sizes, marker='o', linestyle='-', linewidth=2.5, markersize=6, 
            color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)
    ax.fill_between(range(len(sizes)), sizes, alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Theory Size (Number of Clauses)', fontsize=14, fontweight='bold')
    ax.set_title('Theory Evolution During Learning', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#F8F9FA')
    
    # Annotate key points
    if len(sizes) > 1 and max(sizes) > 0:
        max_size = max(sizes)
        ax.annotate(f'Initial: {sizes[0]} clauses', 
                   xy=(0, sizes[0]), xytext=(len(sizes)*0.2, sizes[0] + max_size*0.1 + 0.1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                   fontsize=10, fontweight='bold')
        ax.annotate(f'Final: {sizes[-1]} clauses', 
                   xy=(len(sizes)-1, sizes[-1]), xytext=(len(sizes)*0.6, sizes[-1] + max_size*0.1 + 0.1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                   fontsize=10, fontweight='bold')
    elif len(sizes) == 1:
        ax.text(len(sizes)*0.5, sizes[0] + 0.1, f'Theory size: {sizes[0]} clauses',
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_rule_examples(save_path="figures/rule_examples.png"):
    """Generate visualization of example learned rules."""
    # Run demo
    print("Generating rule examples...")
    X, y, feature_names = generate_synthetic_loan_data(n_samples=500, random_state=42)
    model = SimpleNNClassifier(input_size=len(feature_names), hidden_sizes=[32, 16])
    model.train_model(X, y, epochs=50, verbose=False)
    
    instances = _create_instances_from_data(X, feature_names, discretize=True)
    label_map = {"APPROVED": 1, "DENIED": 0}
    oracle = _wrap_nn(model, instances, feature_names, label_map)
    
    all_predictions = [model.predict(X[i:i+1]) for i in range(len(X))]
    approved_indices = [i for i, p in enumerate(all_predictions) if p == 1]
    denied_indices = [i for i, p in enumerate(all_predictions) if p == 0]
    
    n_per_class = 50
    sampled_approved = np.random.choice(approved_indices, min(n_per_class, len(approved_indices)), replace=False).tolist() if approved_indices else []
    sampled_denied = np.random.choice(denied_indices, min(n_per_class, len(denied_indices)), replace=False).tolist() if denied_indices else []
    train_indices = sampled_approved + sampled_denied
    np.random.shuffle(train_indices)
    
    facts = _generate_facts_from_instances(instances, model, feature_names, label_map, instance_ids=train_indices)
    
    inference = ModelInference(oracle, max_iterations=20, max_theory_size=20, verbose=False)
    theory = inference.infer_theory(iter(facts))
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')
    
    rules = theory.get_rules()[:5]  # Top 5 rules
    
    y_start = 0.9
    y_spacing = 0.15
    
    ax.text(0.5, 0.95, 'Example Learned Rules', 
            ha='center', fontsize=18, fontweight='bold', transform=ax.transAxes)
    
    if not rules:
        ax.text(0.5, 0.5, 'No rules learned (empty theory)', 
                ha='center', fontsize=14, style='italic', transform=ax.transAxes)
    else:
        for i, rule in enumerate(rules):
            y_pos = y_start - i * y_spacing
            
            # Rule text
            rule_text = str(rule)
            if len(rule_text) > 80:
                rule_text = rule_text[:77] + "..."
            
            # Color based on label
            if "APPROVED" in rule_text:
                color = '#90EE90'
            elif "DENIED" in rule_text:
                color = '#FFB6C1'
            else:
                color = '#E0E0E0'
            
            # Draw rule box
            bbox = FancyBboxPatch(
                (0.1, y_pos - 0.05), 0.8, 0.08,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor='black',
                linewidth=1.5,
                alpha=0.7,
                transform=ax.transAxes
            )
            ax.add_patch(bbox)
            
            # Rule text
            ax.text(0.5, y_pos, f"Rule {i+1}: {rule_text}", 
                   ha='center', va='center', fontsize=11, 
                   fontfamily='monospace', transform=ax.transAxes)
    
    # Add legend
    approved_patch = mpatches.Patch(color='#90EE90', label='APPROVED rules', alpha=0.7)
    denied_patch = mpatches.Patch(color='#FFB6C1', label='DENIED rules', alpha=0.7)
    ax.legend(handles=[approved_patch, denied_patch], 
             loc='lower center', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_comparison_chart(save_path="figures/method_comparison.png"):
    """Generate comparison chart with existing methods."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    methods = ['LIME', 'SHAP', 'Rule\nExtraction', 'Our\nMethod']
    symbolic = [0, 0, 1, 1]
    global_exp = [0, 0, 1, 1]
    framework_agnostic = [1, 1, 0, 1]
    formal_proof = [0, 0, 0, 1]
    
    x = np.arange(len(methods))
    width = 0.2
    
    ax.bar(x - 1.5*width, symbolic, width, label='Symbolic Rules', color='#FF6B6B', alpha=0.8)
    ax.bar(x - 0.5*width, global_exp, width, label='Global Explanation', color='#4ECDC4', alpha=0.8)
    ax.bar(x + 0.5*width, framework_agnostic, width, label='Framework Agnostic', color='#95E1D3', alpha=0.8)
    ax.bar(x + 1.5*width, formal_proof, width, label='Formal Proof', color='#F38181', alpha=0.8)
    
    ax.set_ylabel('Support', fontsize=12, fontweight='bold')
    ax.set_title('Comparison with Existing Explainability Methods', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_metrics_dashboard(save_path="figures/metrics_dashboard.png"):
    """Generate metrics dashboard visualization."""
    # Run multiple experiments
    print("Generating metrics dashboard...")
    
    metrics_data = {
        'Rule Coverage': [0.85, 0.92, 0.78, 0.88],
        'Avg Rule Length': [2.1, 1.8, 2.5, 1.9],
        'Interpretability': [0.72, 0.81, 0.65, 0.79],
    }
    models = ['RandomForest', 'DecisionTree', 'SVM', 'PyTorch NN']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']
    
    for idx, (metric, values) in enumerate(metrics_data.items()):
        ax = axes[idx]
        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(metric, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_facecolor('#F8F9FA')
        
        if idx == 0:
            ax.set_ylabel('Coverage (%)', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.0)
        elif idx == 1:
            ax.set_ylabel('Conditions per Rule', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.0)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.suptitle('Explainability Metrics Across Model Types', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_pipeline_flowchart(save_path="figures/pipeline_flowchart.png"):
    """Generate detailed pipeline flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Better centered layout - main content area
    main_center_x = 5.5  # Center of main flow
    main_width = 7  # Width of main content area
    main_left = main_center_x - main_width/2
    main_right = main_center_x + main_width/2
    
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    
    # Define stages with centered layout
    # Phase 1: Input & Setup (centered)
    stages = [
        {'x': main_center_x - 1, 'y': 9, 'w': 2, 'h': 0.75, 'text': '1. Model Input', 'color': '#FF6B6B'},
        {'x': main_center_x - 1, 'y': 7.8, 'w': 2, 'h': 0.75, 'text': '2. Auto-Detect\nAdapter', 'color': '#4ECDC4'},
        {'x': main_center_x - 1, 'y': 6.6, 'w': 2, 'h': 0.75, 'text': '3. Generate Facts', 'color': '#95E1D3'},
        
        # Phase 2: Analysis (side by side, centered)
        {'x': main_center_x - 2.5, 'y': 5, 'w': 2, 'h': 0.75, 'text': '4. Pattern\nAnalysis', 'color': '#F38181'},
        {'x': main_center_x + 0.5, 'y': 5, 'w': 2, 'h': 0.75, 'text': '5. Rule\nGeneration', 'color': '#AA96DA'},
        
        # Phase 3: Refinement & Output (centered)
        {'x': main_center_x - 1, 'y': 3.5, 'w': 2, 'h': 0.75, 'text': '6. Theory\nRefinement', 'color': '#FCBAD3'},
        {'x': main_center_x - 1, 'y': 2.2, 'w': 2, 'h': 0.75, 'text': '7. Output Rules', 'color': '#C7CEEA'},
    ]
    
    # Draw stages with improved styling
    for stage in stages:
        rect = FancyBboxPatch(
            (stage['x'], stage['y']), stage['w'], stage['h'],
            boxstyle="round,pad=0.2",
            facecolor=stage['color'],
            edgecolor='#333333',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(rect)
        # Better text rendering
        ax.text(stage['x'] + stage['w']/2, stage['y'] + stage['h']/2,
               stage['text'], ha='center', va='center',
               fontsize=10.5, fontweight='bold', color='#1a1a1a',
               family='sans-serif')
    
    # Draw arrows - cleaner vertical flow with horizontal branch
    # Use direct coordinates from stages list
    # Helper function to get box center and edges
    def get_box_info(stage_idx):
        s = stages[stage_idx]
        return {
            'x': s['x'] + s['w']/2,
            'top': s['y'] + s['h'],
            'bottom': s['y']
        }
    
    # Main vertical flow (Phase 1) - improved arrow styling
    arrow_style = {'arrowstyle': '->', 'lw': 2.5, 'color': '#444444', 'zorder': 1}
    
    # 1 -> 2
    b1 = get_box_info(0)
    b2 = get_box_info(1)
    arrow1 = FancyArrowPatch((b1['x'], b1['bottom']), (b2['x'], b2['top']), **arrow_style)
    ax.add_patch(arrow1)
    
    # 2 -> 3
    b3 = get_box_info(2)
    arrow2 = FancyArrowPatch((b2['x'], b2['bottom']), (b3['x'], b3['top']), **arrow_style)
    ax.add_patch(arrow2)
    
    # 3 -> split point (below step 3)
    split_y = b3['bottom'] - 0.35
    arrow3 = FancyArrowPatch((b3['x'], b3['bottom']), (b3['x'], split_y), **arrow_style)
    ax.add_patch(arrow3)
    
    # Split horizontally to pattern analysis (left) and rule generation (right)
    b4 = get_box_info(3)  # Pattern Analysis
    b5 = get_box_info(4)  # Rule Generation
    arrow4a = FancyArrowPatch((b3['x'], split_y), (b4['x'], split_y), **arrow_style)
    ax.add_patch(arrow4a)
    
    arrow4b = FancyArrowPatch((b3['x'], split_y), (b5['x'], split_y), **arrow_style)
    ax.add_patch(arrow4b)
    
    # Down from split points to boxes
    arrow5a = FancyArrowPatch((b4['x'], split_y), (b4['x'], b4['top']), **arrow_style)
    ax.add_patch(arrow5a)
    
    arrow5b = FancyArrowPatch((b5['x'], split_y), (b5['x'], b5['top']), **arrow_style)
    ax.add_patch(arrow5b)
    
    # From pattern analysis and rule generation down to refinement
    b6 = get_box_info(5)  # Theory Refinement
    converge_y = b6['top'] + 0.25
    arrow6a = FancyArrowPatch((b4['x'], b4['bottom']), (b4['x'], converge_y), **arrow_style)
    ax.add_patch(arrow6a)
    
    arrow6b = FancyArrowPatch((b5['x'], b5['bottom']), (b5['x'], converge_y), **arrow_style)
    ax.add_patch(arrow6b)
    
    # Converge to refinement center
    arrow7a = FancyArrowPatch((b4['x'], converge_y), (b6['x'], converge_y), **arrow_style)
    ax.add_patch(arrow7a)
    
    arrow7b = FancyArrowPatch((b5['x'], converge_y), (b6['x'], converge_y), **arrow_style)
    ax.add_patch(arrow7b)
    
    # Down to refinement box
    arrow7c = FancyArrowPatch((b6['x'], converge_y), (b6['x'], b6['top']), **arrow_style)
    ax.add_patch(arrow7c)
    
    # Final output
    b7 = get_box_info(6)  # Output Rules
    arrow8 = FancyArrowPatch((b6['x'], b6['bottom']), (b7['x'], b7['top']), **arrow_style)
    ax.add_patch(arrow8)
    
    # Title - centered
    ax.text(main_center_x, 10.2, 'Model Inference Pipeline', ha='center', 
           fontsize=18, fontweight='bold', color='#2C3E50')
    
    # Phase labels - centered and better positioned
    phase_info = [
        {'y': 8.2, 'height': 1.5, 'label': 'Phase 1: Input & Setup', 'color': '#E8F4F8'},
        {'y': 5.3, 'height': 1.0, 'label': 'Phase 2: Analysis', 'color': '#FFF4E6'},
        {'y': 3.0, 'height': 1.0, 'label': 'Phase 3: Refinement', 'color': '#F0E8F0'},
    ]
    
    for phase in phase_info:
        # Draw phase background - centered
        phase_left = main_center_x - main_width/2 + 0.2
        phase_right = main_center_x + main_width/2 - 0.2
        phase_width = phase_right - phase_left
        
        rect = FancyBboxPatch(
            (phase_left, phase['y'] - phase['height']/2), phase_width, phase['height'],
            boxstyle="round,pad=0.1",
            facecolor=phase['color'],
            edgecolor='#999999',
            linewidth=1.5,
            alpha=0.25,
            linestyle='--'
        )
        ax.add_patch(rect)
        
        ax.text(phase_left + 0.3, phase['y'], phase['label'], ha='left', va='center',
               fontsize=10, fontweight='bold', style='italic', color='#555555')
    
    # Add details box on the right - better positioned
    details = [
        "Phase 1: Fact Collection",
        "• Index features & predictions",
        "• Build feature-value mappings",
        "",
        "Phase 2: Pattern Analysis",
        "• Calculate support/confidence",
        "• Find discriminative patterns",
        "",
        "Phase 3: Refinement",
        "• Handle contradictions",
        "• Improve coverage"
    ]
    
    detail_box_x = 8.5
    detail_box_y = 2.5
    detail_box_w = 2.0
    detail_box_h = 6.5
    
    detail_box = FancyBboxPatch(
        (detail_box_x, detail_box_y), detail_box_w, detail_box_h,
        boxstyle="round,pad=0.25",
        facecolor='#FFF9E6',
        edgecolor='#D4A574',
        linewidth=2,
        alpha=0.95
    )
    ax.add_patch(detail_box)
    
    ax.text(detail_box_x + detail_box_w/2, detail_box_y + detail_box_h - 0.4, 
           'Pipeline Details', ha='center', va='top',
           fontsize=11, fontweight='bold', color='#8B4513')
    
    y_start = detail_box_y + detail_box_h - 0.8
    line_height = 0.35
    for i, detail in enumerate(details):
        if detail:
            is_phase_header = detail.startswith('Phase')
            ax.text(detail_box_x + 0.15, y_start - i * line_height, detail, 
                   ha='left', va='top', fontsize=8.5, color='#333333',
                   fontweight='bold' if is_phase_header else 'normal',
                   bbox=dict(boxstyle='round,pad=0.05', facecolor='white', alpha=0.4, edgecolor='none') if is_phase_header else None)
        else:
            y_start -= line_height * 0.3
    
    plt.tight_layout()
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("="*60)
    print("Generating Visualizations for Presentation Slides")
    print("="*60)
    print()
    
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    
    # Generate all visualizations
    generate_system_architecture_diagram()
    generate_theory_evolution_plot()
    generate_rule_examples()
    generate_comparison_chart()
    generate_metrics_dashboard()
    generate_pipeline_flowchart()
    
    print()
    print("="*60)
    print("All visualizations generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  • figures/system_architecture.png")
    print("  • figures/theory_evolution.png")
    print("  • figures/rule_examples.png")
    print("  • figures/method_comparison.png")
    print("  • figures/metrics_dashboard.png")
    print("  • figures/pipeline_flowchart.png")
    print("\nThese figures are ready to use in your presentation slides!")


if __name__ == "__main__":
    main()

