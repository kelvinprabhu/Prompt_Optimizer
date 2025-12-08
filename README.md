# Prompt Optimizer

A comprehensive DSPy-based prompt optimization system for AI/ML tasks, featuring advanced optimization techniques for Retrieval-Augmented Generation (RAG) and Text Summarization.

## 📋 Overview

This project implements sophisticated prompt optimization strategies using DSPy (Declarative Self-improving Language Programs). It includes two main tasks:

1. **Text Summarization**: Optimizes prompts for extracting key information from documents
2. **RAG (Retrieval-Augmented Generation)**: Optimizes prompts for question-answering systems with document retrieval

The project features advanced optimization techniques including:
- Multi-objective optimization with Pareto front analysis
- Genetic algorithm-based prompt mutation
- Diversity-aware candidate pool management
- Comprehensive performance visualization and tracking

## 📁 Project Structure

```
Prompt_Optimizer/
├── DSPy_Basics.ipynb           # Introduction to DSPy fundamentals
├── TASKS/                      # Main task implementations
│   ├── DSpy_PromptOptimizationTask1_RAG.ipynb        # RAG optimization task
│   └── DSpy_PromptOptimizationTask1_Summarizer.ipynb # Summarization optimization task
├── Development/                # Development and experimentation notebooks
│   ├── dspy_g.ipynb           # Genetic algorithm experiments
│   ├── rag.ipynb              # RAG development
│   ├── summarizer.ipynb       # Summarizer development
│   └── [visualization PNGs]   # Generated charts and graphs
├── Outputs/                    # Generated PDF reports
│   ├── DSpy_PromptOptimizationTask1_RAG_*.pdf
│   └── DSpy_PromptOptimizationTask1_Summarizer_*.pdf
├── pdfs/                       # Source documents for testing
│   ├── Book-Summary-Rich-Dad.pdf
│   └── Why Machines Learn PDF.pdf
├── Assignment - AI_ML.pdf      # Assignment documentation
└── README.md                   # This file
```

### Key Directories

- **TASKS/**: Contains the main implementation notebooks for both tasks with complete optimization pipelines
- **Development/**: Experimental notebooks and intermediate development work
- **Outputs/**: Generated reports and analysis PDFs from optimization runs
- **pdfs/**: Sample PDF documents used for RAG and summarization tasks

## 🚀 Features

### 1. Text Summarization Task
- Multi-objective optimization balancing quality, conciseness, and relevance
- Advanced prompt mutation strategies (crossover, replacement, insertion)
- Diversity-aware candidate management
- Comprehensive performance tracking and visualization

### 2. RAG (Retrieval-Augmented Generation) Task
- Vector-based document retrieval using FAISS
- Semantic embeddings with Sentence Transformers
- Context-aware question answering
- Optimized retrieval and generation prompts
- Multi-iteration Pareto optimization

### 3. Optimization Techniques
- **Genetic Algorithms**: Prompt mutation and crossover
- **Multi-Objective Optimization**: Balancing multiple metrics simultaneously
- **Pareto Front Analysis**: Identifying optimal trade-offs
- **Diversity Management**: Maintaining varied candidate pools
- **Iterative Refinement**: Progressive optimization across multiple generations

### 4. Visualizations
The project generates comprehensive visualizations including:
- Optimization trajectory plots
- Pareto front evolution
- Candidate pool diversity analysis
- Merge performance comparisons
- Mutation impact analysis

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Required Dependencies

```bash
# Core dependencies
pip install dspy-ai
pip install numpy pandas
pip install matplotlib seaborn

# NLP and ML libraries
pip install sentence-transformers
pip install scikit-learn
pip install nltk

# Document processing
pip install PyPDF2

# Vector database (for RAG)
pip install faiss-cpu

# LLM providers (optional - choose based on your preference)
pip install langchain-groq langchain-core  # For Groq/Llama models
pip install openai                          # For OpenAI models

# Utilities
pip install tqdm
pip install getpass
```

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/kelvinprabhu/Prompt_Optimizer.git
cd Prompt_Optimizer

# Install all required dependencies
pip install dspy-ai numpy pandas matplotlib seaborn sentence-transformers \
    scikit-learn nltk PyPDF2 faiss-cpu langchain-groq langchain-core tqdm
```

## 🎯 Usage

### Getting Started

1. **Start with DSPy Basics**
   ```bash
   jupyter notebook DSPy_Basics.ipynb
   ```
   This notebook introduces DSPy fundamentals and the custom GroqLM integration.

2. **Run Text Summarization Task**
   ```bash
   jupyter notebook TASKS/DSpy_PromptOptimizationTask1_Summarizer.ipynb
   ```
   - Configure your API key (Groq/OpenAI)
   - Load sample PDF documents
   - Run optimization pipeline
   - View results and visualizations

3. **Run RAG Task**
   ```bash
   jupyter notebook TASKS/DSpy_PromptOptimizationTask1_RAG.ipynb
   ```
   - Set up vector database and embeddings
   - Configure retrieval parameters
   - Execute optimization iterations
   - Analyze Pareto fronts and performance metrics

### API Key Configuration

The notebooks require an LLM API key. You'll be prompted to enter it securely:

```python
import getpass
api_key = getpass.getpass("Enter your GROQ API key: ")
```

Supported LLM providers:
- **Groq**: Fast inference with Llama models (default in notebooks)
- **OpenAI**: GPT models
- **Other**: Any DSPy-compatible LLM provider

### Customization

#### Modify Optimization Parameters

In the task notebooks, you can customize:

```python
# Number of optimization iterations
num_iterations = 3

# Population size for genetic algorithm
population_size = 10

# Mutation rate
mutation_rate = 0.3

# Metrics weights (for multi-objective optimization)
weights = {
    'quality': 0.4,
    'conciseness': 0.3,
    'relevance': 0.3
}
```

#### Add Custom Metrics

Extend the evaluation framework by adding new metrics:

```python
def custom_metric(output, expected):
    # Your metric implementation
    return score
```

## 📊 Outputs and Results

### Generated Visualizations

The optimization process generates several types of visualizations:

1. **Optimization Trajectory**: Shows metric improvements over iterations
2. **Pareto Front**: Displays trade-offs between competing objectives
3. **Candidate Diversity**: Analyzes prompt variation in the population
4. **Mutation Impact**: Tracks effects of different mutation strategies
5. **Merge Performance**: Compares different prompt combination methods

### PDF Reports

Comprehensive PDF reports are generated in the `Outputs/` directory containing:
- Optimization summary
- Final optimized prompts
- Performance metrics
- Visual analysis

## 🔧 Development

### Experimental Notebooks

The `Development/` directory contains:
- **dspy_g.ipynb**: Genetic algorithm experimentation
- **rag.ipynb**: RAG system development
- **summarizer.ipynb**: Summarization system development

These notebooks are used for testing new features and approaches before integration into main tasks.

## 📚 Key Concepts

### DSPy Framework
DSPy provides a declarative approach to building LLM applications with:
- Modular components (Signatures, Modules, Optimizers)
- Automatic prompt optimization
- Type-safe LLM interactions

### Multi-Objective Optimization
Balances competing objectives (e.g., quality vs. conciseness):
- Uses Pareto dominance to identify optimal solutions
- Maintains a diverse set of non-dominated candidates
- Enables informed trade-off decisions

### Genetic Algorithms for Prompts
Applies evolutionary principles to prompt engineering:
- **Mutation**: Random modifications to explore prompt space
- **Crossover**: Combines successful prompts
- **Selection**: Keeps best-performing variants
- **Diversity**: Maintains variety to avoid local optima

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional optimization algorithms
- New task implementations
- Enhanced visualization tools
- Performance optimizations
- Documentation improvements

## 📄 License

This project is part of an AI/ML assignment. Refer to the assignment documentation for specific usage guidelines.

## 🔗 References

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Groq API](https://groq.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)

## 📧 Contact

For questions or issues, please refer to the assignment documentation or open an issue in the repository.

---

**Note**: Make sure to keep your API keys secure and never commit them to version control. Use environment variables or secure key management systems for production deployments.
