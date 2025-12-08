# Prompt Optimizer

A comprehensive DSPy-based prompt optimization framework that implements advanced techniques for optimizing prompts in both Retrieval-Augmented Generation (RAG) and Summarization tasks.

## Overview

This project implements a sophisticated prompt optimization system using DSPy (Declarative Self-improving Language Programs) to automatically improve prompts through multi-objective optimization. The system employs genetic algorithms with mutation, crossover, and diversity-preserving strategies to find optimal prompts that balance multiple objectives like accuracy, diversity, and efficiency.

## Features

- **Multi-Objective Optimization**: Implements Pareto-front optimization to balance multiple competing objectives
- **Genetic Algorithm**: Uses advanced genetic operators including mutation and crossover with diversity preservation
- **Two Main Tasks**:
  - **RAG (Retrieval-Augmented Generation)**: Question-answering system with context retrieval
  - **Summarization**: Document summarization with quality metrics
- **Comprehensive Visualization**: Generates detailed plots tracking optimization progress
- **Custom LM Integration**: Supports GROQ API and other language model providers

## Repository Structure

```
Prompt_Optimizer/
├── Development/                    # Development notebooks and experiments
│   ├── dspy_g.ipynb               # Main genetic algorithm optimization notebook
│   ├── dspy_g copy.ipynb          # Backup/alternative version
│   ├── rag.ipynb                  # RAG development notebook
│   ├── summarizer.ipynb           # Summarizer development notebook
│   ├── candidate_pool_diversity.png
│   ├── merge_performance.png
│   ├── mutation_history.png
│   ├── optimization_trajectory.png
│   └── pareto_front.png
│
├── TASKS/                          # Main task implementations
│   ├── DSpy_PromptOptimizationTask1_RAG.ipynb
│   ├── DSpy_PromptOptimizationTask1_Summarizer.ipynb
│   └── [Various visualization outputs]
│
├── Outputs/                        # Generated PDF reports
│   ├── DSpy_PromptOptimizationTask1_RAG_*.pdf
│   └── DSpy_PromptOptimizationTask1_Summarizer_*.pdf
│
├── pdfs/                          # Source documents for testing
│   ├── Book-Summary-Rich-Dad.pdf
│   └── Why Machines Learn PDF.pdf
│
├── DSPy_Basics.ipynb              # Introduction to DSPy framework
├── Assignment - AI_ML.pdf         # Project assignment details
├── Assignment - AI_ML.docx        # Project assignment (editable)
└── README.md                      # This file
```

### Directory Details

#### `/Development`
Contains experimental and development notebooks:
- **dspy_g.ipynb**: Primary notebook implementing genetic algorithm-based prompt optimization
- **rag.ipynb**: Development work for RAG system
- **summarizer.ipynb**: Development work for summarization task
- **Visualization outputs**: PNG files showing optimization metrics

#### `/TASKS`
Contains the final implementation notebooks for the two main tasks:
- **DSpy_PromptOptimizationTask1_RAG.ipynb**: Complete RAG implementation with prompt optimization
- **DSpy_PromptOptimizationTask1_Summarizer.ipynb**: Complete summarization implementation with prompt optimization
- Includes comprehensive visualizations of optimization progress

#### `/Outputs`
Contains generated PDF reports documenting the optimization results and performance metrics.

#### `/pdfs`
Source PDF documents used for testing RAG and summarization tasks.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Jupyter Notebook or JupyterLab
- GROQ API key (or other supported LLM provider)

### Required Dependencies

Install the required packages using pip:

```bash
pip install dspy-ai
pip install PyPDF2
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install sentence-transformers
pip install nltk
pip install faiss-cpu
pip install langchain-groq
pip install langchain-core
pip install tqdm
```

Or install all at once:

```bash
pip install dspy-ai PyPDF2 numpy pandas matplotlib seaborn scikit-learn sentence-transformers nltk faiss-cpu langchain-groq langchain-core tqdm
```

## Usage

### Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kelvinprabhu/Prompt_Optimizer.git
   cd Prompt_Optimizer
   ```

2. **Set Up Environment**
   - Install dependencies as listed above
   - Obtain a GROQ API key from [GROQ](https://console.groq.com/)

3. **Run Notebooks**
   
   Start with the basics:
   ```bash
   jupyter notebook DSPy_Basics.ipynb
   ```

### Running the Main Tasks

#### Task 1: Summarization
Open and run `TASKS/DSpy_PromptOptimizationTask1_Summarizer.ipynb`:
- Implements prompt optimization for document summarization
- Uses genetic algorithms to improve summary quality
- Generates visualizations of optimization progress
- Exports results to PDF

#### Task 2: RAG (Retrieval-Augmented Generation)
Open and run `TASKS/DSpy_PromptOptimizationTask1_RAG.ipynb`:
- Implements optimized RAG pipeline for question answering
- Uses FAISS for efficient vector similarity search
- Optimizes prompts for better retrieval and generation
- Includes comprehensive evaluation metrics

### Development Notebooks

The `/Development` folder contains experimental notebooks where new optimization strategies are tested:
- `dspy_g.ipynb`: Main genetic algorithm implementation
- `rag.ipynb`: RAG system development
- `summarizer.ipynb`: Summarization system development

## Key Components

### 1. Prompt Optimization Engine

The system uses a genetic algorithm approach with:
- **Mutation**: Random variations in prompts to explore the search space
- **Crossover**: Combining successful prompts to create new candidates
- **Diversity Preservation**: Maintains a diverse pool of solutions
- **Pareto Front**: Multi-objective optimization balancing competing goals

### 2. GroqLM Integration

Custom DSPy language model integration for GROQ API:
```python
class GroqLM(dspy.LM):
    # Custom implementation for GROQ API
    # Supports both prompt and messages format
```

### 3. RAG System

- PDF document processing and chunking
- Sentence embedding using SentenceTransformers
- FAISS vector database for efficient retrieval
- Context-aware answer generation

### 4. Summarization System

- Document preprocessing and cleaning
- Quality metrics (ROUGE, coherence, relevance)
- Iterative prompt improvement
- Comprehensive evaluation framework

## Visualization Outputs

The system generates several types of visualizations:

1. **Pareto Front**: Shows the trade-off between competing objectives
2. **Optimization Trajectory**: Tracks improvement over iterations
3. **Mutation History**: Visualizes the impact of mutations
4. **Candidate Pool Diversity**: Measures diversity of prompt candidates
5. **Merge Performance**: Analyzes crossover strategy effectiveness

## Workflow

1. **Initialize**: Set up language model and base prompt
2. **Generate Candidates**: Create initial pool of prompt variations
3. **Evaluate**: Score candidates on multiple objectives
4. **Select**: Choose best performers using Pareto optimization
5. **Evolve**: Apply mutation and crossover operations
6. **Iterate**: Repeat until convergence or max iterations
7. **Export**: Generate reports and visualizations

## Configuration

Key parameters that can be adjusted:

- **Population Size**: Number of prompt candidates in each generation
- **Mutation Rate**: Probability of random prompt modifications
- **Crossover Rate**: Frequency of combining successful prompts
- **Iterations**: Number of optimization cycles
- **Objectives**: Metrics to optimize (accuracy, diversity, efficiency)

## Output Files

- **PDF Reports**: Comprehensive documentation of optimization results
- **PNG Visualizations**: Charts and graphs showing optimization metrics
- **JSON Logs**: Detailed logs of optimization process (if enabled)

## API Keys and Environment Variables

Create a `.env` file or use `getpass` to securely input your API keys:

```python
import getpass
api_key = getpass.getpass("Enter your GROQ API key: ")
```

Note: Never commit API keys to version control. The `.env` file is already in `.gitignore`.

## Contributing

This is an academic project. For improvements or suggestions:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Technical Details

### DSPy Framework
This project leverages DSPy (Declarative Self-improving Language Programs), a framework that:
- Automatically optimizes prompts and weights
- Provides modular components for LLM applications
- Supports various optimization techniques
- Enables reproducible experiments

### Optimization Approach
The genetic algorithm implementation includes:
- **Fitness Function**: Multi-objective evaluation combining accuracy, diversity, and efficiency
- **Selection Strategy**: Tournament selection with elitism
- **Mutation Operators**: Various techniques to introduce controlled randomness
- **Crossover Operators**: Multiple strategies for combining successful prompts

## Troubleshooting

### Common Issues

1. **API Rate Limits**: If you hit rate limits, reduce batch size or add delays
2. **Memory Issues**: For large documents, increase chunk size or reduce batch processing
3. **Model Availability**: Ensure your GROQ API key has access to required models

### Dependencies Issues

If you encounter import errors:
```bash
pip install --upgrade dspy-ai
pip install --upgrade sentence-transformers
```

## License

This project is part of an academic assignment. Please refer to your institution's guidelines for usage and distribution.

## Acknowledgments

- **DSPy Framework**: For providing the optimization infrastructure
- **GROQ**: For LLM API access
- **Sentence Transformers**: For embedding models
- **FAISS**: For efficient similarity search

## Contact

For questions or issues, please open an issue in the GitHub repository or contact the repository owner.

---

**Note**: This project is designed for educational purposes to demonstrate prompt optimization techniques using genetic algorithms and multi-objective optimization in the context of RAG and summarization tasks.
