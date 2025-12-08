"""
Example script showing what the pipeline flowchart looks like when generated.
This creates a simplified version to demonstrate the output.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

def create_example_flowchart():
    """Create an example flowchart showing the pipeline structure."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Centered layout
    center_x = 5.5
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    
    # Define stages
    stages = [
        {'x': center_x - 1, 'y': 9, 'w': 2, 'h': 0.75, 'text': '1. Model Input', 'color': '#FF6B6B'},
        {'x': center_x - 1, 'y': 7.8, 'w': 2, 'h': 0.75, 'text': '2. Auto-Detect\nAdapter', 'color': '#4ECDC4'},
        {'x': center_x - 1, 'y': 6.6, 'w': 2, 'h': 0.75, 'text': '3. Generate Facts', 'color': '#95E1D3'},
        {'x': center_x - 2.5, 'y': 5, 'w': 2, 'h': 0.75, 'text': '4. Pattern\nAnalysis', 'color': '#F38181'},
        {'x': center_x + 0.5, 'y': 5, 'w': 2, 'h': 0.75, 'text': '5. Rule\nGeneration', 'color': '#AA96DA'},
        {'x': center_x - 1, 'y': 3.5, 'w': 2, 'h': 0.75, 'text': '6. Theory\nRefinement', 'color': '#FCBAD3'},
        {'x': center_x - 1, 'y': 2.2, 'w': 2, 'h': 0.75, 'text': '7. Output Rules', 'color': '#C7CEEA'},
    ]
    
    # Draw boxes
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
        ax.text(stage['x'] + stage['w']/2, stage['y'] + stage['h']/2,
               stage['text'], ha='center', va='center',
               fontsize=10.5, fontweight='bold', color='#1a1a1a')
    
    # Draw arrows
    arrow_style = {'arrowstyle': '->', 'lw': 2.5, 'color': '#444444', 'zorder': 1}
    
    def get_box_info(idx):
        s = stages[idx]
        return {'x': s['x'] + s['w']/2, 'top': s['y'] + s['h'], 'bottom': s['y']}
    
    # Vertical flow
    b1, b2, b3 = get_box_info(0), get_box_info(1), get_box_info(2)
    ax.add_patch(FancyArrowPatch((b1['x'], b1['bottom']), (b2['x'], b2['top']), **arrow_style))
    ax.add_patch(FancyArrowPatch((b2['x'], b2['bottom']), (b3['x'], b3['top']), **arrow_style))
    
    # Split
    split_y = b3['bottom'] - 0.35
    ax.add_patch(FancyArrowPatch((b3['x'], b3['bottom']), (b3['x'], split_y), **arrow_style))
    
    b4, b5 = get_box_info(3), get_box_info(4)
    ax.add_patch(FancyArrowPatch((b3['x'], split_y), (b4['x'], split_y), **arrow_style))
    ax.add_patch(FancyArrowPatch((b3['x'], split_y), (b5['x'], split_y), **arrow_style))
    ax.add_patch(FancyArrowPatch((b4['x'], split_y), (b4['x'], b4['top']), **arrow_style))
    ax.add_patch(FancyArrowPatch((b5['x'], split_y), (b5['x'], b5['top']), **arrow_style))
    
    # Converge
    b6 = get_box_info(5)
    converge_y = b6['top'] + 0.25
    ax.add_patch(FancyArrowPatch((b4['x'], b4['bottom']), (b4['x'], converge_y), **arrow_style))
    ax.add_patch(FancyArrowPatch((b5['x'], b5['bottom']), (b5['x'], converge_y), **arrow_style))
    ax.add_patch(FancyArrowPatch((b4['x'], converge_y), (b6['x'], converge_y), **arrow_style))
    ax.add_patch(FancyArrowPatch((b5['x'], converge_y), (b6['x'], converge_y), **arrow_style))
    ax.add_patch(FancyArrowPatch((b6['x'], converge_y), (b6['x'], b6['top']), **arrow_style))
    
    # Final
    b7 = get_box_info(6)
    ax.add_patch(FancyArrowPatch((b6['x'], b6['bottom']), (b7['x'], b7['top']), **arrow_style))
    
    # Title
    ax.text(center_x, 10.2, 'Model Inference Pipeline', ha='center', 
           fontsize=18, fontweight='bold', color='#2C3E50')
    
    # Phase labels
    phases = [
        {'y': 8.2, 'h': 1.5, 'label': 'Phase 1: Input & Setup', 'color': '#E8F4F8'},
        {'y': 5.3, 'h': 1.0, 'label': 'Phase 2: Analysis', 'color': '#FFF4E6'},
        {'y': 3.0, 'h': 1.0, 'label': 'Phase 3: Refinement', 'color': '#F0E8F0'},
    ]
    
    for phase in phases:
        phase_left = center_x - 3.3
        phase_width = 6.6
        rect = FancyBboxPatch(
            (phase_left, phase['y'] - phase['h']/2), phase_width, phase['h'],
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
    
    # Details box
    details_box = FancyBboxPatch(
        (8.5, 2.5), 2.0, 6.5,
        boxstyle="round,pad=0.25",
        facecolor='#FFF9E6',
        edgecolor='#D4A574',
        linewidth=2,
        alpha=0.95
    )
    ax.add_patch(details_box)
    
    ax.text(9.5, 8.8, 'Pipeline Details', ha='center', va='top',
           fontsize=11, fontweight='bold', color='#8B4513')
    
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
    
    y_start = 8.5
    for i, detail in enumerate(details):
        if detail:
            is_header = detail.startswith('Phase')
            ax.text(8.65, y_start - i * 0.35, detail, ha='left', va='top',
                   fontsize=8.5, color='#333333', fontweight='bold' if is_header else 'normal',
                   bbox=dict(boxstyle='round,pad=0.05', facecolor='white', alpha=0.4) if is_header else None)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/example_flowchart.png', bbox_inches='tight', facecolor='white', dpi=300)
    print("✓ Example flowchart saved to: figures/example_flowchart.png")
    print("\nFlowchart Structure:")
    print("=" * 60)
    print("Title: Model Inference Pipeline")
    print("\nPhase 1: Input & Setup")
    print("  → 1. Model Input (Red)")
    print("  → 2. Auto-Detect Adapter (Teal)")
    print("  → 3. Generate Facts (Light Green)")
    print("\nPhase 2: Analysis")
    print("  → 4. Pattern Analysis (Pink) ←→ 5. Rule Generation (Purple)")
    print("\nPhase 3: Refinement")
    print("  → 6. Theory Refinement (Light Pink)")
    print("  → 7. Output Rules (Light Blue)")
    print("\nRight Side: Pipeline Details box with phase descriptions")
    print("=" * 60)
    plt.close()

if __name__ == "__main__":
    create_example_flowchart()



