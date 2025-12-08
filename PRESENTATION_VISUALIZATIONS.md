# Presentation Visualizations

This document describes the visualizations generated for your presentation slides.

## Generated Files

All visualizations are saved in the `figures/` directory:

1. **`system_architecture.png`** (174 KB)
   - Shows the plug-and-play explainability pipeline
   - Displays the flow from model input through adapters, observation languages, oracle, inference algorithm, to final rules and explanations

2. **`theory_evolution.png`** (155 KB)
   - Visualizes how the theory size grows during learning
   - Shows iterations on x-axis and number of clauses on y-axis
   - Includes annotations for initial and final theory sizes

3. **`rule_examples.png`** (174 KB)
   - Displays example learned rules from the loan approval classifier
   - Color-coded by label (APPROVED in green, DENIED in pink)
   - Shows up to 5 example rules

4. **`method_comparison.png`** (99 KB)
   - Comparison chart showing how our method compares to LIME, SHAP, and Rule Extraction
   - Shows support for: Symbolic Rules, Global Explanation, Framework Agnostic, Formal Proof

5. **`metrics_dashboard.png`** (194 KB)
   - Three-panel dashboard showing metrics across different model types
   - Metrics: Rule Coverage, Average Rule Length, Interpretability Score
   - Models: RandomForest, DecisionTree, SVM, PyTorch NN

6. **`pipeline_flowchart.png`** (252 KB)
   - Detailed 7-step pipeline flowchart
   - Shows the three phases: Fact Collection, Pattern Analysis, Theory Refinement
   - Includes annotations explaining each phase

## Usage

### Running the Script

```bash
python generate_slide_visualizations.py
```

This will generate all 6 visualizations in the `figures/` directory.

### Individual Functions

You can also call individual functions:

```python
from generate_slide_visualizations import (
    generate_system_architecture_diagram,
    generate_theory_evolution_plot,
    generate_rule_examples,
    generate_comparison_chart,
    generate_metrics_dashboard,
    generate_pipeline_flowchart
)

# Generate specific visualization
generate_system_architecture_diagram("custom_path.png")
```

## Slide Integration

These visualizations are designed to be inserted directly into your presentation slides:

- **System Architecture**: Use in slides explaining the overall system design
- **Theory Evolution**: Use in slides showing the learning process
- **Rule Examples**: Use in slides demonstrating learned rules
- **Method Comparison**: Use in slides comparing with existing methods
- **Metrics Dashboard**: Use in slides showing experimental results
- **Pipeline Flowchart**: Use in slides explaining the methodology

## Customization

To customize the visualizations:

1. Edit the color schemes in each function
2. Modify figure sizes by changing `figsize` parameters
3. Adjust fonts and styling via matplotlib rcParams
4. Change data sources to use your own experimental results

## Notes

- All figures are saved at 300 DPI for high-quality printing
- Figures use a consistent color scheme across all visualizations
- The script handles edge cases (empty theories, missing directories, etc.)
- Visualizations use real data from your codebase when possible

## Troubleshooting

If you encounter issues:

1. **Import errors**: Make sure you're running from the project root directory
2. **Empty theories**: The script handles empty theories gracefully
3. **File permissions**: Ensure you have write permissions in the `figures/` directory
4. **Missing dependencies**: Install required packages: `pip install matplotlib seaborn numpy`

## Next Steps

1. Review the generated visualizations
2. Insert them into your presentation slides
3. Adjust colors/styling to match your presentation theme if needed
4. Regenerate with updated data if you have new experimental results



