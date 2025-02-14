
# MASTER-FastToG Integrated System

This repository implements an integrated reasoning system that combines the strengths of two cutting-edge approaches:

- **MASTER: A Multi-Agent System with LLM Specialized MCTS**  
  ([PDF](https://arxiv.org/pdf/2501.14304))  
  MASTER leverages a multi-agent Monte Carlo Tree Search (MCTS) framework to guide and refine the reasoning process of large language models (LLMs) through repeated self-evaluation, selection, expansion, and backpropagation.

- **Fast Think-on-Graph (FastToG): Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph**  
  ([PDF](https://arxiv.org/pdf/2501.14300))  
  FastToG enhances LLM reasoning by retrieving and summarizing structured subgraphs from a knowledge graph. This graph-to-text conversion provides rich, contextually relevant information to ground the LLM's output.

## Overview

The integrated system combines MASTER’s sophisticated multi-agent MCTS planning with FastToG’s efficient, graph-based context enrichment. In this framework:

- **Graph Retrieval:** A FastToG component retrieves and summarizes relevant subgraph information (e.g., via BFS or DFS) around a target entity from a knowledge graph. This summary is injected into the agent's prompt.
- **Multi-Agent MCTS:** MASTER-style agents generate partial solutions, validate and assess them, and backpropagate scores if a terminal solution fails evaluation.
- **Dynamic Context Augmentation:** Child agents inherit the parent's reasoning context along with updated graph-derived knowledge if new entities emerge during reasoning.

The result is a robust, knowledge-grounded reasoning pipeline that improves both the efficiency and accuracy of LLM-based problem solving.

## Features

- **Multi-Step LLM Reasoning:** Agents call the LLM multiple times to generate a solution, perform validation, assess with a numeric score and confidence, and conduct a final evaluation.
- **Graph-Based Context Enrichment:** Integrates FastToG methods to convert a knowledge graph substructure into a textual summary that enhances each agent’s prompt.
- **Multi-Agent MCTS Planning:** Uses a modified UCT algorithm for agent selection, expansion, and backpropagation to focus on promising reasoning paths.
- **Scalability and Efficiency:** Parallelized agent expansion and dynamic context updates reduce token consumption and improve inference speed.

## Requirements

- Python 3.8+
- Python packages: `requests`, `numpy`, `igraph`, `matplotlib`, etc.
- Access to an LLM API (e.g., OpenAI GPT-4)  
- Access to a knowledge graph database (e.g., Neo4j) for FastToG retrieval

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/multi-agent-MCTS-FastToG.git
cd MASTER-FastToG
```

## Usage

To run the system with a sample query, use:

```bash
python main.py
```

You can modify the query and target entity in `main.py` to test different scenarios.

## Architecture

The system is organized into several modules:

- **Knowledge Graph Retrieval (FastToG):**  
  Retrieves a subgraph around a target entity and converts it into a textual summary (via BFS, DFS, or MST methods).

- **MASTER MCTS Agent:**  
  Implements the multi-agent reasoning framework where each agent generates a solution, validates, and assesses it. Agents expand by creating child agents, and the system uses a modified UCT formula for selection and backpropagation.

- **Controller:**  
  Orchestrates the overall reasoning process by integrating graph context with LLM calls, managing agent selection, expansion, and evaluation.

## Citations

If you use this code or the ideas in your research, please cite:

1. **MASTER: A Multi-Agent System with LLM Specialized MCTS**  
   [https://arxiv.org/pdf/2501.14304](https://arxiv.org/pdf/2501.14304)

2. **Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph**  
   [https://arxiv.org/pdf/2501.14300](https://arxiv.org/pdf/2501.14300)

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgments

Special thanks to the authors of the MASTER and FastToG papers for their innovative work on enhancing LLM reasoning with multi-agent planning and knowledge graph-based context enrichment.

