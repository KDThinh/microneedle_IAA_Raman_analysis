# Documentation Files Comparison

This document explains the differences between the three main documentation files in this project.

---

## Quick Summary

| File | Purpose | Audience | Focus |
|------|---------|----------|-------|
| **PROCESSING_GUIDE.md** | **How to use** the pipeline | End users | Practical operations |
| **DATA_PROCESSING_STEPS.md** | **What happens** in the pipeline | Developers/analysts | Workflow documentation |
| **PROCESSING_ALGORITHM_ANALYSIS.md** | **Why and how well** algorithms work | Researchers/developers | Algorithm evaluation |

---

## 1. PROCESSING_GUIDE.md (User Manual)

**Purpose**: Practical guide for running the pipeline

**Content**:
- How to create and modify profiles in `pipeline.yml`
- Command-line usage examples
- How to run different entry points (`main_updated_v1.py`, `main_wo_append.py`, etc.)
- Troubleshooting common issues
- Auto-generating profiles with scripts

**Audience**: 
- End users who need to process data
- New users learning the system
- Anyone who wants to run the pipeline

**Key Questions It Answers**:
- "How do I process my data?"
- "What command do I run?"
- "How do I create a profile?"
- "What do I do if I get an error?"

**Length**: ~146 lines (shortest, most practical)

---

## 2. DATA_PROCESSING_STEPS.md (Workflow Documentation)

**Purpose**: Detailed step-by-step documentation of the entire processing pipeline

**Content**:
- Complete workflow from start to finish
- What happens at each processing stage:
  1. Configuration and Data Loading
  2. Spectral Processing (Single Scan)
  3. Batch Spectral Processing
  4. Plotting
  5. Post-Processing (ALS, Diurnal, FFT)
  6. Master CSV Updates
  7. Vertical Distribution Plots
- Parameter summaries
- Output files generated
- Technical details of each step

**Audience**:
- Developers understanding the codebase
- Analysts who need to understand data transformations
- Anyone debugging or modifying the pipeline

**Key Questions It Answers**:
- "What happens to my data at each step?"
- "What parameters are used where?"
- "What files are generated?"
- "What's the sequence of operations?"

**Length**: ~325 lines (medium length, comprehensive workflow)

---

## 3. PROCESSING_ALGORITHM_ANALYSIS.md (Algorithm Evaluation)

**Purpose**: Critical analysis and evaluation of the algorithms used in the pipeline

**Content**:
- Detailed analysis of each algorithm:
  - Savitzky-Golay Filter
  - Lieberfit (Iterative Polynomial Baseline)
  - ALS (Asymmetric Least Squares)
  - Gaussian Smoothing
  - Fast Fourier Transform (FFT)
- Strengths and weaknesses of each method
- Recommendations for improvements
- Algorithm complexity analysis
- Validation recommendations
- Comparison with alternative methods

**Audience**:
- Researchers evaluating methodology
- Developers considering algorithm improvements
- Anyone doing algorithm research or optimization

**Key Questions It Answers**:
- "Why was this algorithm chosen?"
- "What are the limitations?"
- "How can we improve the processing?"
- "What alternatives exist?"
- "Is the implementation optimal?"

**Length**: ~420 lines (longest, most technical)

---

## When to Use Each Document

### Use PROCESSING_GUIDE.md when:
- ✅ You're new to the project
- ✅ You want to process data quickly
- ✅ You need to troubleshoot a runtime issue
- ✅ You're setting up a new experiment profile

### Use DATA_PROCESSING_STEPS.md when:
- ✅ You need to understand the data flow
- ✅ You're debugging processing issues
- ✅ You want to know what parameters affect what
- ✅ You're documenting or explaining the pipeline
- ✅ You need to understand intermediate outputs

### Use PROCESSING_ALGORITHM_ANALYSIS.md when:
- ✅ You're evaluating algorithm choices
- ✅ You're planning improvements to the pipeline
- ✅ You need to justify methodology in publications
- ✅ You're comparing with other processing methods
- ✅ You're optimizing parameters

---

## Overlap and Relationships

**PROCESSING_GUIDE.md** and **DATA_PROCESSING_STEPS.md**:
- Both cover the pipeline workflow
- Guide focuses on "how to run it"
- Steps focuses on "what it does"

**DATA_PROCESSING_STEPS.md** and **PROCESSING_ALGORITHM_ANALYSIS.md**:
- Both cover technical details
- Steps focuses on "what happens"
- Analysis focuses on "why and how well"

**PROCESSING_GUIDE.md** and **PROCESSING_ALGORITHM_ANALYSIS.md**:
- Minimal overlap
- Guide is practical, Analysis is theoretical
- Guide is for users, Analysis is for developers/researchers

---

## Recommended Reading Order

1. **Start with PROCESSING_GUIDE.md** - Learn how to use the system
2. **Then read DATA_PROCESSING_STEPS.md** - Understand what's happening
3. **Finally read PROCESSING_ALGORITHM_ANALYSIS.md** - Deep dive into algorithms (optional, for advanced users)

---

## Summary

- **PROCESSING_GUIDE.md** = "User Manual" (practical, operational)
- **DATA_PROCESSING_STEPS.md** = "Technical Documentation" (workflow, detailed)
- **PROCESSING_ALGORITHM_ANALYSIS.md** = "Research Paper" (evaluation, recommendations)

All three documents serve different purposes and complement each other. Keep all three for a complete documentation set.

