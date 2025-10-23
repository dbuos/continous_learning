# Qwen Continuous Learning Experiments

**Model Tested:** Qwen/Qwen2.5-0.5B-Instruct (494M parameters)
**Date:** 2025-10-23

## Overview

This subdirectory contains experiments validating soft prompting continuous learning with Qwen2.5 models. We tested the same soft prompting approach from the parent project but with Qwen's architecture and instruction-tuned models.

## Key Findings

### Experiment 1: With Chat Template ✓ (Recommended)

**Configuration:**
- Model: Qwen/Qwen2.5-0.5B-Instruct
- Soft prompt tokens: 8 (7,168 trainable parameters)
- Training episodes: 80
- Learning rate: 0.001
- Chat template: **Enabled**

**Results:**
- **Initial accuracy: 65.0%** (vs TinyLlama's 20%)
- **Final accuracy: 62.0%**
- **Improvement: -3.0%** (slight decrease, within noise)
- **Speed: 1.60 episodes/sec**

**Analysis:**
- Qwen models are much stronger out-of-the-box due to superior instruction tuning
- Starting at 65% accuracy leaves less room for improvement
- Minimal degradation suggests soft prompting doesn't harm performance
- Per-task performance:
  - **Best:** Causal reasoning (83.3% final)
  - **Good:** Math word problems (72.7%)
  - **Challenging:** Sequence prediction (33.3%)

### Experiment 2: Without Chat Template

**Configuration:**
- Same as above, but **chat template disabled**

**Results:**
- **Initial accuracy: 10.0%** (model struggles without proper formatting)
- **Final accuracy: 20.0%**
- **Improvement: +10.0%** (shows learning is happening)
- **Speed: 0.57 episodes/sec** (slower due to longer generation)

**Analysis:**
- Without chat template, model produces incoherent outputs
- Soft prompting helps but can't fully compensate for bad formatting
- Demonstrates importance of proper prompt engineering

## Comparison: Qwen vs TinyLlama

| Metric | Qwen (w/ chat) | TinyLlama | Winner |
|--------|----------------|-----------|---------|
| Model size | 494M | 1.1B | TinyLlama |
| Initial accuracy | **65%** | 20% | **Qwen** |
| Final accuracy | **62%** | 40% | **Qwen** |
| Improvement | -3% | **+20%** | **TinyLlama** |
| Speed (eps/sec) | 1.60 | 0.47 | **Qwen** |

**Key Insights:**
1. **Qwen is better pre-trained:** 65% vs 20% initial accuracy
2. **TinyLlama shows more improvement:** +20% vs -3%
3. **Qwen is faster:** 3.4x more episodes/sec
4. **Ceiling effects:** Qwen starts so high that soft prompting doesn't help much

## Architecture Adaptations

The Qwen learner (`qwen_learner.py`) includes:

1. **Chat template support:** Leverages Qwen's instruction format
   ```python
   messages = [
       {"role": "system", "content": "You are a helpful assistant..."},
       {"role": "user", "content": question}
   ]
   ```

2. **Improved answer extraction:** Handles both simple and chat formats
   ```python
   if "assistant" in generated_text.lower():
       answer = parts[-1].strip()
   ```

3. **Same soft prompting core:** 8 learnable embedding vectors prepended to inputs

## File Structure

```
qwen_experiments/
├── README.md                           # This file
├── qwen_learner.py                     # Qwen-specific continuous learner
├── train_qwen.py                       # Training script with Qwen support
├── reasoning_arena.py                  # Same task generator as parent
├── checkpoints/                        # Experiment 1 (with chat template)
│   ├── checkpoint_ep20.pt
│   ├── checkpoint_ep40.pt
│   ├── checkpoint_ep60.pt
│   ├── checkpoint_ep80.pt
│   ├── results.json
│   └── learning_curves.png
└── checkpoints_no_template/            # Experiment 2 (without chat template)
    ├── checkpoint_ep20.pt
    ├── checkpoint_ep40.pt
    ├── checkpoint_ep60.pt
    ├── checkpoint_ep80.pt
    ├── results.json
    └── learning_curves.png
```

## Running Experiments

### Basic Training
```bash
python train_qwen.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --episodes 80 \
    --eval-interval 20 \
    --soft-tokens 8 \
    --lr 0.001
```

### Without Chat Template
```bash
python train_qwen.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --episodes 80 \
    --no-chat-template
```

### Other Qwen Models
```bash
# Larger model (more capable but slower)
python train_qwen.py --model Qwen/Qwen2.5-1.5B-Instruct

# Smaller model (faster but less capable)
python train_qwen.py --model Qwen/Qwen2.5-0.5B
```

## Conclusions

### What Works
✅ Soft prompting is architecturally compatible with Qwen models
✅ Chat templates dramatically improve baseline performance
✅ Qwen models are fast and efficient for continuous learning
✅ No catastrophic forgetting or instability observed

### What Doesn't Work
❌ Limited improvement when baseline is already strong (ceiling effect)
❌ Without chat template, performance is too poor to be useful
❌ Sequence prediction remains challenging across all models

### Recommendations

1. **For production use:** Use Qwen with chat template enabled
2. **For learning demonstrations:** Use TinyLlama (shows clearer improvement)
3. **For efficiency:** Qwen is 3.4x faster per episode
4. **For accuracy:** Qwen provides better absolute performance

## Future Work

1. **Test larger Qwen models:** Qwen2.5-1.5B, Qwen2.5-3B
2. **Hybrid approaches:** Combine soft prompting with few-shot examples
3. **Task-specific prompts:** Different soft prompts for different task types
4. **Curriculum learning:** Start with easy tasks, gradually increase difficulty
5. **Meta-learning:** Learn to adapt soft prompts faster

## References

- Qwen2.5 Models: https://huggingface.co/Qwen
- Parent project: `/content/` (TinyLlama experiments)
- Original paper: "The Power of Scale for Parameter-Efficient Prompt Tuning" (Lester et al., 2021)
