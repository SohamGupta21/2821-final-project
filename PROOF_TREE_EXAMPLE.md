# Proof Tree Example - Framework Output

This document shows an example of a **proof tree** that is generated as output from the framework when it proves a prediction using learned rules.

## What is a Proof Tree?

A proof tree shows **how the framework logically derives a prediction** by:
1. Starting with a goal (e.g., `predict(instance_0, APPROVED)`)
2. Finding matching rules that can prove the goal
3. Proving the rule's conditions (subgoals) recursively
4. Using facts when conditions are ground truths

## Example: Proving Loan Approval

### Input Theory (Learned Rules + Facts)

**Rules:**
```
Rule 1: predict(X, APPROVED) :- 
    feature(X, income, high), 
    feature(X, credit_score, high).

Rule 2: predict(X, DENIED) :- 
    feature(X, credit_score, low).
```

**Facts:**
```
feature(instance_0, income, high).
feature(instance_0, credit_score, high).
feature(instance_0, age, medium).
```

### Goal to Prove
```
predict(instance_0, APPROVED)
```

### Proof Tree Structure

```
                    predict(instance_0, APPROVED)
                              │
                              │ [Rule 1]
                              │
                    ┌─────────┴─────────┐
                    │                   │
    feature(instance_0, income, high)   feature(instance_0, credit_score, high)
                    │                   │
                    │ [Fact]            │ [Fact]
                    │                   │
                    ✓                   ✓
```

### Text Representation

```
├── predict(instance_0, APPROVED)
│   └── [RULE] predict(X, APPROVED) :- 
│       feature(X, income, high), 
│       feature(X, credit_score, high).
│   ├── feature(instance_0, income, high)
│   │   └── [FACT] feature(instance_0, income, high).
│   └── feature(instance_0, credit_score, high)
│       └── [FACT] feature(instance_0, credit_score, high).
```

## Visual Elements

The proof tree visualization uses color coding:

- **Green boxes**: Facts (ground truths from the data)
- **Blue boxes**: Rules (learned Horn clauses)
- **Yellow boxes**: Goals (what we're trying to prove)

## How It Works

1. **Root Goal**: Start with `predict(instance_0, APPROVED)`

2. **Find Matching Rule**: Rule 1 matches because:
   - Head: `predict(X, APPROVED)` unifies with goal
   - Substitution: `X = instance_0`

3. **Prove Subgoals**: Must prove both conditions:
   - `feature(instance_0, income, high)` → Found as fact ✓
   - `feature(instance_0, credit_score, high)` → Found as fact ✓

4. **Success**: All subgoals proven → Goal is proven!

## Example: Proving Denial

### Goal
```
predict(instance_1, DENIED)
```

### Facts
```
feature(instance_1, credit_score, low).
```

### Proof Tree
```
predict(instance_1, DENIED)
    │
    │ [Rule 2]
    │
feature(instance_1, credit_score, low)
    │
    │ [Fact]
    │
    ✓
```

## Usage

To generate proof tree examples:

```bash
python generate_proof_tree_example.py
```

This creates:
- `figures/example_proof_tree.png` - Visual proof tree
- Console output showing the proof structure

## Real-World Application

In the actual framework:

1. **Model makes prediction**: e.g., "APPROVED" for instance_0
2. **Framework learns rules**: Extracts patterns like Rule 1
3. **Framework proves prediction**: Shows WHY it's APPROVED using the proof tree
4. **Explanation**: "Instance approved because income=high AND credit_score=high"

This provides **interpretable, provable explanations** for model predictions!



