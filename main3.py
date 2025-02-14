import math
import random
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import concurrent.futures
import openai
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --------------------------------------------
# Logging configuration
# --------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------
# Initialize OpenAI client (Set your API key)
# --------------------------------------------
client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")  # Replace with your actual API key


# --------------------------------------------
# SECTION A: Knowledge Graph Retrieval (FastToG)
# --------------------------------------------

class graph2text_client:
    def __init__(self, model_path, max_length: int = 128):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.max_length = max_length

    def generate(self, triples: str, max_length=None) -> str:
        if len(triples) == 0:
            return ''
        TASK_PREFIX = 'triple to text: '  # Define TASK_PREFIX here
        inputs = self.tokenizer(TASK_PREFIX + triples, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length if max_length is not None else self.max_length,
            do_sample=False  # disable sampling to test if batching affects output
        ).cpu()
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[-1]


class KnowledgeGraph:
    """
    A simple knowledge graph representation using in-memory data structures.
    """

    def __init__(self):
        self.nodes: Dict[int, str] = {}  # Node ID to label
        self.edges: Dict[int, List[Tuple[int, str]]] = defaultdict(list)  # Node ID to (neighbor ID, relation)
        self.node_counter: int = 0
        self.g2t_client = graph2text_client(model_path="YOUR_FASTTOG_MODEL_PATH") # Replace

    def add_node(self, label: str) -> int:
        """Adds a node to the graph and returns its ID."""
        if label not in self.nodes.values():
            node_id = self.node_counter
            self.nodes[node_id] = label
            self.node_counter += 1
            return node_id
        else:  # Node already exists, return existing ID
            return [k for k, v in self.nodes.items() if v == label][0]

    def add_edge(self, subject_label: str, relation: str, object_label: str):
        """Adds an edge to the graph.  Adds nodes if they don't exist."""
        subject_id = self.add_node(subject_label)
        object_id = self.add_node(object_label)
        self.edges[subject_id].append((object_id, relation))

    def get_neighbors(self, node_id: int, max_neighbors: int = 10) -> List[Tuple[int, str]]:
      """Retrieves neighbors of a node, limiting the number of neighbors."""
      return self.edges.get(node_id, [])[:max_neighbors]

    def bfs_summarize(self, start_label: str, max_hops: int = 3, max_neighbors: int = 5) -> str:
        """
        Performs BFS from start_label and returns a textual summary using FastToG.

        Args:
            start_label: The label of the starting node.
            max_hops: Maximum number of hops for BFS.
            max_neighbors: Maximum neighbors to consider at each step.

        Returns:
            A textual summary of the traversed subgraph.
        """
        start_id = self.add_node(start_label) # Ensure node exists, get ID
        if start_id not in self.nodes:
          return f"Node with label '{start_label}' not found in the graph."

        visited = set()
        queue = deque([(start_id, 0)])  # (node_id, hop_count)
        visited.add(start_id)
        triples = []

        while queue and max_hops > 0:
            current_id, hop_count = queue.popleft()

            if hop_count >= max_hops:  # Check hop count *before* fetching neighbors
              continue

            neighbors = self.get_neighbors(current_id, max_neighbors=max_neighbors)
            for neighbor_id, relation in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, hop_count + 1))
                    triples.append(
                        f"({self.nodes[current_id]}, {relation}, {self.nodes[neighbor_id]})"
                    )
        
        # Use FastToG to generate a summary
        triples_str = " ".join(triples)
        if not triples_str:
            return f"No information found within {max_hops} hops of '{start_label}'."

        summary = self.g2t_client.generate(triples_str)
        return summary


# In-memory cache for graph retrieval results.
graph_cache: Dict[str, str] = {}

def fetch_graph_info(entity: str, kg: KnowledgeGraph, max_hops: int = 3, max_neighbors: int = 5) -> str:
    """
    Retrieve and summarize the subgraph around the given entity.
    Uses caching and the KnowledgeGraph instance.
    """
    key = entity.lower()
    if key in graph_cache:
        logger.info(f"Graph info for '{entity}' retrieved from cache.")
        return graph_cache[key]

    summary = kg.bfs_summarize(entity, max_hops=max_hops, max_neighbors=max_neighbors)
    graph_cache[key] = summary
    logger.info(f"Graph info for '{entity}' cached.")
    return summary



# --------------------------------------------
# SECTION B: LLM Multi-Step Reasoning (with error handling)
# --------------------------------------------

def call_gpt_api(prompt: str, model: str = "gpt-4o", max_tokens: int = 200) -> str:
    """
    Calls the GPT-4o API using OpenAI's Python SDK.  Handles errors.
    Allows specifying the model and max_tokens.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return "API_ERROR"
    except Exception as e:  # Catch other potential errors
        logger.error(f"An unexpected error occurred: {e}")
        return "API_EXCEPTION"


def llm_generate_solution(prompt: str) -> str:
    result = call_gpt_api(f"[Solve]\n{prompt}")
    return result

def llm_validate(solution: str) -> str:
    result = call_gpt_api(f"[Validate]\n{solution}")
    return result

def llm_assess(solution: str, validation_text: str) -> Tuple[float, float]:
    """Assess the solution and validation, returning score and confidence."""
    prompt = f"[Assess]\nSolution: {solution}\nValidation: {validation_text}"
    result = call_gpt_api(prompt, max_tokens=50)  # Shorter max_tokens for assessment

    try:
        # Attempt to parse score and confidence.  More robust parsing.
        parts = result.split()
        score = float(parts[0])
        confidence = float(parts[1]) if len(parts) > 1 else 0.5 # Default confidence
        return score, confidence

    except (ValueError, IndexError) as e:
        logger.warning(f"Parsing LLM assess output failed: {e}; using random values.")
        return random.uniform(0, 1), random.uniform(0.1, 1.0)

def llm_evaluate(solution: str) -> bool:
    """Evaluate the final solution."""
    result = call_gpt_api(f"[Final Evaluation]\n{solution}", max_tokens=50)
    if result == "API_ERROR" or result == "API_EXCEPTION":
        # If the API fails, we can't evaluate, so return False (or perhaps a random choice)
        return False

    return "correct" in result.lower()


# --------------------------------------------
# SECTION C: MASTER-MCTS Agent Implementation
# --------------------------------------------

class Agent:
    def __init__(self, prompt: str, parent: Optional['Agent'] = None):
        self.prompt = prompt
        self.parent = parent
        self.children: List['Agent'] = []
        self.solution: str = ""
        self.validation: str = ""
        self.score: float = 0.0  # r0
        self.confidence: float = 0.0  # c0
        self.value: float = 0.0
        self.visits: int = 0
        self.is_terminal: bool = False
        self.evaluation_passed: bool = False

    def run_llm_calls(self):
        logger.info(f"Agent running LLM calls with prompt: {self.prompt[:50]}...")
        self.solution = llm_generate_solution(self.prompt)
        if self.solution.startswith("API"):
          return
        self.validation = llm_validate(self.solution)
        if self.validation.startswith("API"):
          return

        self.score, self.confidence = llm_assess(self.solution, self.validation)
        # Terminal criteria:  Based on confidence or score.
        self.is_terminal = self.confidence > 0.8 or self.score > 0.9
        if self.is_terminal:
            self.evaluation_passed = llm_evaluate(self.solution)
        logger.info(f"Agent produced solution: {self.solution[:50]}... Terminal: {self.is_terminal}, Eval Passed: {self.evaluation_passed}")

    def expand(self, kg: KnowledgeGraph) -> Tuple[bool, str, float]:
        child_prompt = self.build_child_prompt(kg)
        child_agent = Agent(child_prompt, parent=self)
        self.children.append(child_agent)
        child_agent.run_llm_calls()
        return (child_agent.evaluation_passed, child_agent.solution, child_agent.score)

    def build_child_prompt(self, kg: KnowledgeGraph) -> str:
        """
        Builds the prompt for the child agent, incorporating parent context and KG info.
        """
        base = f"[Parent Prompt]: {self.prompt}\n[Parent Solution]: {self.solution}"

        # Extract entities from the parent's solution and fetch KG info.
        # This is a very basic entity extraction; use a proper NER system in practice.
        entities = [word for word in self.solution.split() if word[0].isupper()]
        for entity in entities:
            extra_info = fetch_graph_info(entity, kg)
            if extra_info and "No specific graph info" not in extra_info:
                base += f"\n[Extra KG Info for {entity}]:\n{extra_info}"
        return base

    def backpropagate(self, reward: float):
        self.visits += 1
        # Use a simple average for the value update.
        self.value = (self.value * (self.visits - 1) + reward) / self.visits
        logger.debug(f"Backpropagated reward {reward}; New value: {self.value}, Visits: {self.visits}")

        if self.parent:  # Recursively backpropagate to the parent
            self.parent.backpropagate(reward)

def compute_uct(agent: Agent, parent_visits: int, exploration_constant: float = 1.0) -> float:
    """Compute the UCT value for an agent."""
    if agent.visits == 0:
        return float('inf')  # Encourage exploration of unvisited nodes
    exploitation_term = agent.confidence * agent.score + (1 - agent.confidence) * agent.value
    exploration_term = exploration_constant * math.sqrt(math.log(parent_visits) / agent.visits)
    return exploitation_term + exploration_term


def select_with_uct(root: Agent) -> Agent:
    """Selects the best child agent using the UCT formula."""
    current = root
    while current.children:
        best_child = max(current.children, key=lambda child: compute_uct(child, current.visits))
        current = best_child
    return current

def find_best_agent(root: Agent) -> Agent:
    """Finds the agent with the highest value (not UCT) in the tree."""
    best_agent = root
    max_value = root.value

    stack = [root]
    while stack:
        agent = stack.pop()
        if agent.value > max_value:
            max_value = agent.value
            best_agent = agent
        stack.extend(agent.children)  # Corrected line

    return best_agent


# --------------------------------------------
# SECTION D: MASTER+FastToG Controller
# --------------------------------------------

class MASTERFastToG:
    def __init__(self, question: str, entity: str, max_expansion: int, num_branches: int):
        self.kg = KnowledgeGraph()  # Initialize the Knowledge Graph
        # Add some initial data to the graph (for demonstration)
        self.kg.add_edge("Pennsylvania Convention Center", "located_in", "Philadelphia")
        self.kg.add_edge("Philadelphia", "has_climate", "Humid Subtropical Climate")
        self.kg.add_edge("Philadelphia", "state", "Pennsylvania")
        self.kg.add_edge("Pennsylvania", "climate_type", "Humid Continental Climate")


        # Retrieve KG context using FastToG
        graph_text = fetch_graph_info(entity, self.kg)  # Pass the kg instance
        combined_prompt = f"Question: {question}\n[KG Context]:\n{graph_text}"
        logger.info(f"Combined prompt for root agent:\n{combined_prompt}")

        self.root = Agent(combined_prompt)
        self.root.run_llm_calls()
        self.max_expansion = max_expansion
        self.num_branches = num_branches

    def run(self) -> str:
        if self.root.is_terminal and self.root.evaluation_passed:
            return f"Final Answer: {self.root.solution}"

        for _ in range(self.max_expansion):
            selected = select_with_uct(self.root)
            logger.info(f"Selected agent with prompt: {selected.prompt[:50]}... (visits: {selected.visits})")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(selected.expand, self.kg) for _ in range(self.num_branches)]  # Pass kg
                for future in concurrent.futures.as_completed(futures):
                    child_eval_passed, child_solution, child_score = future.result()
                    logger.info(f"Child expansion: Solution: {child_solution[:50]}..., Eval Passed: {child_eval_passed}")
                    if child_eval_passed:
                         return f"Final Answer: {child_solution}"
                    # Backpropagate only the score
                    selected.backpropagate(child_score)  # Only backpropagate the score
        best = find_best_agent(self.root)
        return f"Best Attempt: {best.solution}"

# --------------------------------------------
# SECTION E: Example Usage
# --------------------------------------------

if __name__ == "__main__":
    question = "What kind of clothing should I bring if I visit the Pennsylvania Convention Center in spring?"
    entity = "Pennsylvania Convention Center"
    max_expansion = 4
    num_branches = 2

    system = MASTERFastToG(question, entity, max_expansion, num_branches)
    answer = system.run()
    logger.info(f"System final answer: {answer}")
    print(answer)
