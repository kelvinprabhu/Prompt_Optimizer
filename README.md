# DSPy Prompt Optimization Framework

A comprehensive implementation of **GEPA (Generate, Evaluate, Pareto-filter, Adapt)** prompt optimization using DSPy for two key NLP tasks: Text Summarization and Retrieval-Augmented Generation (RAG).

## 📋 Overview

This project demonstrates advanced prompt optimization techniques using the DSPy framework and the GEPA methodology to automatically improve prompt quality through evolutionary optimization with multi-objective Pareto filtering.

### Key Features

- **GEPA Optimization Framework**: Multi-objective prompt optimization using Generate-Evaluate-Pareto-Adapt cycles
- **Text Summarization**: Optimized prompts for extractive and abstractive summarization
- **RAG System**: Retrieval-Augmented Generation with optimized retrieval and generation strategies
- **Comprehensive Visualizations**: Detailed charts showing optimization trajectories, Pareto fronts, and performance metrics
- **LLM Integration**: Uses Groq's LLama 3.3 70B model via DSPy

## 🗂️ Project Structure

```
Prompt_Optimizer/
│
├── TASKS/                                           # Main implementation notebooks
│   ├── DSpy_PromptOptimizationTask1_Summarizer.ipynb   # Text summarization optimization
│   ├── DSpy_PromptOptimizationTask1_RAG.ipynb          # RAG system optimization
│   └── *.png                                            # Generated visualization outputs
│
├── Development/                                     # Development and testing notebooks
│   ├── summarizer.ipynb                            # Summarizer prototypes
│   ├── rag.ipynb                                   # RAG prototypes
│   └── dspy_g.ipynb                                # DSPy experimentation
│
├── pdfs/                                           # PDF documents for testing
│   ├── Book-Summary-Rich-Dad.pdf                   # Test document for summarization
│   └── Why Machines Learn PDF.pdf                  # Test document for RAG
│
├── Outputs/                                        # Exported PDF outputs
│   ├── DSpy_PromptOptimizationTask1_RAG_*.pdf
│   └── DSpy_PromptOptimizationTask1_Summarizer_*.pdf
│
├── DSPy_Basics.ipynb                               # DSPy framework basics and examples
├── .env                                            # Environment variables (API keys)
└── README.md                                       # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
conda create -n gepa-opt python=3.10
conda activate gepa-opt
```

### Installation

```bash
# Install required packages
pip install dspy-ai PyPDF2 numpy pandas matplotlib seaborn scikit-learn
pip install sentence-transformers faiss-cpu langchain-groq langchain-core tqdm
```

### API Key Setup

Create a `.env` file or set your API key:

```python
import getpass
api_key = getpass.getpass("Enter your GROQ API key: ")
```

Get your GROQ API key from: https://console.groq.com/

## 📚 Tasks Overview

### Task 1: Text Summarizer Optimization

**Notebook**: `TASKS/DSpy_PromptOptimizationTask1_Summarizer.ipynb`

Implements GEPA optimization for text summarization with:

- **Candidate Pool Initialization**: Diverse prompt candidates with varying styles, tones, and approaches
- **Multi-Objective Evaluation**: ROUGE scores, diversity, coverage, and fluency metrics
- **Pareto Filtering**: Select non-dominated candidates based on multiple objectives
- **Reflective Mutation**: Intelligent prompt mutations based on performance analysis
- **System-Aware Merging**: Combine best candidates for improved prompts

**Key Components**:
- `PromptCandidate`: Dataclass storing prompt templates and metadata
- `CandidatePoolInitializer`: Generate diverse initial prompt pool
- `ParetoFilter`: Multi-objective optimization using Pareto dominance
- `ReflectiveMutator`: Analyze and mutate prompts based on weaknesses
- `SystemAwareMerger`: Merge top-performing candidates
- `GEPASummarizer`: Complete GEPA-optimized summarization system

**Metrics**:
- ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)
- Compression ratio
- Diversity score
- Coverage score

### Task 2: RAG System Optimization

**Notebook**: `TASKS/DSpy_PromptOptimizationTask1_RAG.ipynb`

Implements GEPA optimization for RAG with:

- **Vector Index**: FAISS-based semantic search with sentence transformers
- **RAG Pipeline**: Retrieval → Context formatting → LLM generation
- **Multi-Component Optimization**: Optimizes both retrieval and generation prompts
- **Evaluation Metrics**: Retrieval recall, generation F1, faithfulness scores

**Key Components**:
- `VectorIndex`: FAISS vector store for document retrieval
- `RAGPromptCandidate`: Stores retrieval and generation instructions
- `BaselineRAG`: Standard RAG implementation
- `GEPARAG`: GEPA-optimized RAG system
- `RAGParetoFilter`: Multi-objective filtering for RAG candidates
- `RAGReflectiveMutator`: Intelligent mutations for RAG prompts

**Metrics**:
- Retrieval recall
- Generation F1 score
- Faithfulness to context
- Answer coverage

## 🔬 GEPA Methodology

The GEPA framework follows a 4-step evolutionary optimization cycle:

### 1. **Generate** - Candidate Pool Initialization
```python
# Create diverse prompt candidates
candidates = pool_initializer.generate_initial_pool()
```

- Generate 12-20 diverse prompt candidates
- Vary instruction style, tone, format, and emphasis
- Measure diversity using cosine similarity

### 2. **Evaluate** - Performance Measurement
```python
# Evaluate each candidate on multiple metrics
for candidate in candidates:
    metrics = evaluate_candidate(candidate, test_data)
```

- Multi-objective evaluation (ROUGE, F1, faithfulness, etc.)
- Track performance across different dimensions
- Aggregate scores for ranking

### 3. **Pareto-filter** - Multi-Objective Selection
```python
# Select non-dominated candidates
pareto_front = pareto_filter.get_pareto_front(candidates)
```

- Identify Pareto-optimal candidates
- Keep solutions that excel in different objectives
- Maintain diversity in the solution space

### 4. **Adapt** - Mutation and Merging
```python
# Mutate based on performance analysis
mutated = mutator.mutate_candidate(candidate)

# Merge top candidates
merged = merger.merge_candidates(pareto_front, top_k=3)
```

- Reflective mutation based on weakness analysis
- System-aware merging of top performers
- Generate new candidates for next iteration

## 📊 Visualizations

Both tasks generate comprehensive visualizations:

### Summarizer Visualizations
- `candidate_pool_diversity.png` - Similarity matrix of initial candidates
- `pareto_front.png` - 2D/3D Pareto frontier visualization
- `mutation_history.png` - Impact analysis of mutations
- `merge_performance.png` - Merging strategy effectiveness
- `optimization_trajectory.png` - Score progression over iterations

### RAG Visualizations
- `rag_candidate_diversity.png` - RAG candidate pool diversity
- `rag_pareto_front_iter*.png` - Pareto fronts per iteration
- `rag_mutation_impact.png` - Mutation effectiveness
- `rag_merge_strategy.png` - Merge performance
- `rag_optimization_trajectory.png` - Complete optimization progress
- `rag_comprehensive_comparison.png` - Baseline vs GEPA comparison

## 🎯 Results

### Summarizer Performance
- **Baseline ROUGE-L**: ~0.35
- **GEPA-Optimized ROUGE-L**: ~0.42
- **Improvement**: +20% in summary quality

### RAG Performance
- **Baseline Combined Score**: 0.65
- **GEPA Combined Score**: 0.82
- **Improvement**: +26% in overall performance

## 🔧 Customization

### Adding New Tasks

1. Define your task signature:
```python
class MyTaskSignature(dspy.Signature):
    input_field = dspy.InputField(desc="...")
    output_field = dspy.OutputField(desc="...")
```

2. Create a candidate class:
```python
@dataclass
class MyPromptCandidate:
    instruction: str
    format: str
    score: float = 0.0
```

3. Implement GEPA components:
- Pool initializer
- Evaluation metrics
- Pareto filter
- Mutator and merger

### Tuning Hyperparameters

```python
# Adjust optimization parameters
gepa_system = GEPAOptimizer(
    pool_size=12,          # Number of initial candidates
    n_iterations=3,        # Optimization cycles
    top_k=3,              # Candidates to merge
    objectives=['metric1', 'metric2', 'metric3']
)
```

## 📖 Key Concepts

### DSPy Framework
- **Signatures**: Define input/output specifications
- **Modules**: Composable LLM components
- **Predictions**: Structured outputs from LLMs

### Pareto Optimization
- **Dominance**: Solution A dominates B if better in all objectives
- **Pareto Front**: Set of non-dominated solutions
- **Trade-offs**: Balance between competing objectives

### Prompt Engineering
- **Chain-of-Thought**: Step-by-step reasoning
- **Few-Shot Learning**: Examples in prompts
- **Meta-Prompting**: Prompts that generate prompts

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional optimization algorithms (e.g., genetic algorithms)
- More evaluation metrics
- Support for other LLM providers
- Automated hyperparameter tuning
- Multi-task optimization

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **DSPy Framework**: Stanford NLP Group
- **GEPA Methodology**: Adapted from research in prompt optimization
- **Groq**: Fast LLM inference platform

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: Make sure to keep your API keys secure and never commit them to version control. Use environment variables or `.env` files (included in `.gitignore`).
