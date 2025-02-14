import math
import random
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import concurrent.futures
import openai
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import spacy  # Import spaCy

# --------------------------------------------
# Logging configuration
# --------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------
# Constants
# --------------------------------------------
MAX_EXPANSION = 4
NUM_BRANCHES = 2
EXPLORATION_CONSTANT = 1.0
CONFIDENCE_THRESHOLD = 0.8
SCORE_THRESHOLD = 0.9
FASTTOG_MODEL_PATH = "YOUR_FASTTOG_MODEL_PATH"  # REPLACE WITH YOUR MODEL PATH
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"       # REPLACE WITH YOUR API KEY

# --------------------------------------------
# Initialize OpenAI client and spaCy
# --------------------------------------------
client = openai.OpenAI(api_key=OPENAI_API_KEY)
nlp = spacy.load("en_core_web_sm")  # Load a small spaCy English model

# --------------------------------------------
# SECTION A: Knowledge Graph Retrieval (FastToG)
# --------------------------------------------

class graph2text_client:
    def __init__(self, model_path, max_length: int = 128):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.max_length = max_length
        self.task_prefix = 'triple to text: ' # Initialize task prefix

    def generate(self, triples: str, max_length=None) -> str:
        if not triples:
            return ''
        try:
            inputs = self.tokenizer(self.task_prefix + triples, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length if max_length is not None else self.max_length,
                do_sample=False
            ).cpu()
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[-1]
        except Exception as e:
            logger.error(f"FastToG generation error: {e}")
            return "FastToG generation failed."  # Return a fallback message



class KnowledgeGraph:
    """
    A simple knowledge graph representation using in-memory data structures.
    """

    def __init__(self):
        self.nodes: Dict[int, str] = {}  # Node ID to label
        self.edges: Dict[int, List[Tuple[int, str]]] = defaultdict(list)  # Node ID to (neighbor ID, relation)
        self.node_counter: int = 0
        try:
          self.g2t_client = graph2text_client(model_path=FASTTOG_MODEL_PATH)
        except Exception as e:
          logger.error(f"Failed to initialize FastToG client: {e}")
          self.g2t_client = None # Set to None if initialization fails.

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
        """
        start_id = self.add_node(start_label)
        if start_id not in self.nodes:
            return f"Node with label '{start_label}' not found."

        visited = set()
        queue = deque([(start_id, 0)])  # (node_id, hop_count)
        visited.add(start_id)
        triples = []

        while queue and max_hops > 0:
            current_id, hop_count = queue.popleft()

            if hop_count >= max_hops:
                continue

            neighbors = self.get_neighbors(current_id, max_neighbors=max_neighbors)
            for neighbor_id, relation in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, hop_count + 1))
                    triples.append(
                        f"({self.nodes[current_id]}, {relation}, {self.nodes[neighbor_id]})"
                    )

        triples_str = " ".join(triples)
        if not triples_str:
            return f"No information found within {max_hops} hops of '{start_label}'."

        if self.g2t_client: # Check if g2t_client is initialized
          summary = self.g2t_client.generate(triples_str)
        else:
          summary = "FastToG client not initialized.  Could not generate summary."
        return summary


# In-memory cache for graph retrieval results.
graph_cache: Dict[str, str] = {}

def fetch_graph_info(entity: str, kg: KnowledgeGraph, max_hops: int = 3, max_neighbors: int = 5) -> str:
    """Retrieve and summarize subgraph."""
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
    """Calls the GPT-4o API with error handling."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return "API_ERROR"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "API_EXCEPTION"


def llm_generate_solution(prompt: str) -> str:
    return call_gpt_api(f"[Solve]\n{prompt}")

def llm_validate(solution: str) -> str:
    return call_gpt_api(f"[Validate]\n{solution}")

def llm_assess(solution: str, validation_text: str) -> Tuple[float, float]:
    prompt = f"[Assess]\nSolution: {solution}\nValidation: {validation_text}"
    result = call_gpt_api(prompt, max_tokens=50)
    if result.startswith("API"):  # Handle API errors directly
        return 0.0, 0.0

    try:
        parts = result.split()
        score = float(parts[0])
        confidence = float(parts[1]) if len(parts) > 1 else 0.5
        return score, confidence
    except (ValueError, IndexError) as e:
        logger.warning(f"Parsing LLM assess output failed: {e}; using default values.")
        return 0.0, 0.5  # Return default values on parsing failure

def llm_evaluate(solution: str) -> bool:
    result = call_gpt_api(f"[Final Evaluation]\n{solution}", max_tokens=50)
    if result.startswith("API"):
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
        self.score: float = 0.0
        self.confidence: float = 0.0
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
        self.is_terminal = self.confidence > CONFIDENCE_THRESHOLD or self.score > SCORE_THRESHOLD
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
        """Builds the prompt for the child agent."""
        base = f"[Parent Prompt]: {self.prompt}\n[Parent Solution]: {self.solution}"

        # Use spaCy for entity extraction
        doc = nlp(self.solution)
        entities = [ent.text for ent in doc.ents]

        for entity in entities:
            extra_info = fetch_graph_info(entity, kg)
            if extra_info and "No specific graph info" not in extra_info:
                base += f"\n[Extra KG Info for {entity}]:\n{extra_info}"

        # Add a prompt for the child to refine the solution or ask a sub-question
        base += "\n[Refine or ask a sub-question based on the above information]"
        return base

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value = (self.value * (self.visits - 1) + reward) / self.visits
        logger.debug(f"Backpropagated reward {reward}; New value: {self.value}, Visits: {self.visits}")

        if self.parent:
            self.parent.backpropagate(reward)

def compute_uct(agent: Agent, parent_visits: int, exploration_constant: float = EXPLORATION_CONSTANT) -> float:
    """Compute the UCT value for an agent."""
    if agent.visits == 0:
        return float('inf')
    exploitation_term = agent.confidence * agent.score + (1 - agent.confidence) * agent.value
    exploration_term = exploration_constant * math.sqrt(math.log(parent_visits) / agent.visits)
    return exploitation_term + exploration_term


def select_with_uct(root: Agent) -> Agent:
    """Selects the best child agent using UCT."""
    current = root
    while current.children:
        best_child = max(current.children, key=lambda child: compute_uct(child, current.visits))
        current = best_child
    return current

def find_best_agent(root: Agent) -> Agent:
    """Finds the agent with the highest value."""
    best_agent = root
    max_value = root.value

    stack = [root]
    while stack:
        agent = stack.pop()
        if agent.value > max_value:
            max_value = agent.value
            best_agent = agent
        stack.extend(agent.children)

    return best_agent


# --------------------------------------------
# SECTION D: MASTER+FastToG Controller
# --------------------------------------------

class MASTERFastToG:
    def __init__(self, question: str, entity: str, max_expansion: int = MAX_EXPANSION, num_branches: int = NUM_BRANCHES):
        self.kg = KnowledgeGraph()
        # Add initial data to the graph
        self.kg.add_edge("Pennsylvania Convention Center", "located_in", "Philadelphia")
        self.kg.add_edge("Philadelphia", "has_climate", "Humid Subtropical Climate")
        self.kg.add_edge("Philadelphia", "state", "Pennsylvania") #add state relation
        self.kg.add_edge("Pennsylvania", "climate_type", "Humid Continental Climate")


        graph_text = fetch_graph_info(entity, self.kg)
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
                futures = [executor.submit(selected.expand, self.kg) for _ in range(self.num_branches)]
                for future in concurrent.futures.as_completed(futures):
                    child_eval_passed, child_solution, child_score = future.result()
                    logger.info(f"Child expansion: Solution: {child_solution[:50]}..., Eval Passed: {child_eval_passed}")
                    if child_eval_passed:
                        return f"Final Answer: {child_solution}"
                    selected.backpropagate(child_score)

        best = find_best_agent(self.root)
        return f"Best Attempt: {best.solution}"

# --------------------------------------------
# SECTION E: Example Usage
# --------------------------------------------

if __name__ == "__main__":
    question = "What kind of clothing should I bring if I visit the Pennsylvania Convention Center in spring?"
    entity = "Pennsylvania Convention Center"


    system = MASTERFastToG(question, entity)
    answer = system.run()
    logger.info(f"System final answer: {answer}")
    print(answer)
