import math
import random
from collections import deque
from typing import Optional, List, Tuple

##############################################
# SECTION A: Dummy Community & FastToG Retrieval
##############################################
class Community:
    """
    A minimal dummy Community class.
    In practice, import Community (and related functions)
    from your community_tool.py.
    """
    def __init__(self, ent_triples: List[Tuple[int, str, int]], rel_triples: List[Tuple]):
        self.ent_triples = ent_triples  # List of tuples: (node_id, label, distance)
        self.rel_triples = rel_triples  # List of relation tuples; dummy for now

    def bfs2text(self) -> List[str]:
        """
        Dummy BFS: If only one node, returns an empty list.
        Otherwise, returns a list of dummy edge descriptions.
        """
        if len(self.ent_triples) <= 1:
            return []
        # In a real system, you would traverse your subgraph.
        return [f"({self.ent_triples[0][1]}, dummy_relation, {self.ent_triples[1][1]})"]

def fasttog_retrieve(question: str, entity: str) -> str:
    """
    Simulates FastToG's retrieval pipeline:
      1. Identify a relevant community/subgraph for `entity`.
      2. Convert that subgraph into text (using BFS/DFS).
      3. Return the text to enrich the LLM prompt.
    """
    # For demonstration, we create a dummy community with one node.
    dummy_community = Community(
        ent_triples=[(0, entity, 0)],
        rel_triples=[]
    )
    text_lines = dummy_community.bfs2text()
    if not text_lines:
        text_lines = [f"No edges found. Node: {entity}"]
    graph_text = "\n".join(text_lines)
    return f"[FastToG Graph Context]\n{graph_text}"


##############################################
# SECTION B: MASTER-Like Agent & MCTS
##############################################
class Agent:
    """
    Represents a single agent in the MASTER multi-agent MCTS.
    Each agent holds a prompt (which can be enriched with graph context),
    and goes through multiple steps: generating a solution, validation,
    assessment (score & confidence), and a final evaluation if terminal.
    """
    def __init__(self, prompt: str, parent: Optional['Agent'] = None):
        self.prompt = prompt
        self.parent = parent
        self.children: List[Agent] = []
        
        # LLM output fields (in practice, these are results of LLM API calls)
        self.thought: str = ""
        self.action: str = ""
        self.observation: str = ""
        self.validation: str = ""
        self.assessment: str = ""  # textual assessment summary
        self.score: float = 0.0    # initial reward (r0)
        self.confidence: float = 0.0  # confidence (c0)
        
        # MCTS backprop fields
        self.value: float = 0.0
        self.visits: int = 0
        
        # Terminal flag and final evaluation
        self.is_terminal: bool = False
        self.evaluation_passed: bool = False

    def action_procedure(self):
        """
        Simulates the chain of LLM calls:
          - Generates a Thought and Action based on the prompt.
          - Gets an Observation (e.g., from a tool/environment).
          - Performs Validation.
          - Assesses the solution (yielding score and confidence).
          - If terminal, runs a final Evaluation.
        For demo purposes, these are random placeholders.
        """
        self.thought = f"Reasoning about: {self.prompt[:60]}..."
        self.action = f"Proposed solution: based on {self.thought[:40]}"
        self.observation = "Observation: simulated tool output."
        self.validation = "Validation: simulated check."
        self.score = random.uniform(0, 1)
        self.confidence = random.uniform(0.1, 1.0)
        
        # Decide terminality heuristically (replace with real logic)
        self.is_terminal = (random.random() > 0.5)
        if self.is_terminal:
            self.evaluation_passed = (random.random() > 0.4)

    def expand(self) -> Tuple[bool, str, float]:
        """
        Expands the current agent by creating a child agent with an aggregated prompt.
        Returns a tuple: (child_evaluation_passed, child_solution, child_score).
        """
        child_prompt = self._build_child_prompt()
        child_agent = Agent(child_prompt, parent=self)
        self.children.append(child_agent)
        child_agent.action_procedure()
        return (child_agent.evaluation_passed, child_agent.action, child_agent.score)

    def _build_child_prompt(self) -> str:
        """
        Builds the prompt for a child agent by combining the parent's prompt and solution.
        Optionally, new entities in the solution could trigger a new FastToG retrieval.
        """
        lines = [f"[Parent Prompt]: {self.prompt}",
                 f"[Parent Action]: {self.action}"]
        # Example: if parent's action mentions a new entity, add extra context.
        if "Philadelphia" in self.action:
            extra_context = fasttog_retrieve(self.prompt, "Philadelphia")
            lines.append(f"[Extra KG Context]:\n{extra_context}")
        return "\n".join(lines)

    def backpropagate(self, reward: float):
        """
        Updates this agent's value using a running average.
        """
        if self.visits == 0:
            self.value = reward
        else:
            self.value = (self.value * self.visits + reward) / (self.visits + 1)
        self.visits += 1


def compute_uct(agent: Agent, parent_visits: int) -> float:
    """
    Computes the modified UCT value for an agent.
    If agent.visits is 0, returns agent.score.
    Otherwise, combines exploitation and exploration:
      UCT = (c0 * r0) + ((1-c0) * agent.value) + exploration,
      where exploration = (1 / (10 * sqrt(2*c0))) * sqrt(ln(parent_visits) / agent.visits)
    """
    if agent.visits == 0:
        return agent.score
    exploitation = agent.confidence * agent.score + (1 - agent.confidence) * agent.value
    exploration = (1 / (10 * math.sqrt(2 * agent.confidence))) * math.sqrt(math.log(max(parent_visits, 1)) / agent.visits)
    return exploitation + exploration

def select_with_uct(root: Agent) -> Agent:
    """
    Traverses the tree from the root, selecting the child with the highest UCT
    until reaching a leaf agent.
    """
    current = root
    while current.children:
        best_child = None
        best_uct = float('-inf')
        for child in current.children:
            child_uct = compute_uct(child, max(current.visits, 1))
            if child_uct > best_uct:
                best_uct = child_uct
                best_child = child
        if not best_child:
            break
        current = best_child
    return current

def find_best_agent(root: Agent) -> Agent:
    """
    Returns the agent with the highest aggregated value in the entire tree.
    """
    stack = [root]
    best_node = root
    while stack:
        node = stack.pop()
        if node.value > best_node.value:
            best_node = node
        stack.extend(node.children)
    return best_node

##############################################
# SECTION D: MASTER + FastToG Combined System
##############################################
class MASTERFastToG:
    def __init__(self, question: str, entity: str, max_expansion: int, num_branches: int):
        """
        Initializes the system:
          1) Uses FastToG to retrieve KG context for the given entity.
          2) Combines the question with KG context to build the root prompt.
          3) Creates the root agent and runs its LLM calls.
        """
        graph_text = fasttog_retrieve(question, entity)
        combined_prompt = f"Question: {question}\n{graph_text}"
        self.root = Agent(combined_prompt)
        self.root.action_procedure()
        self.max_expansion = max_expansion
        self.num_branches = num_branches

    def run(self) -> str:
        """
        Runs the MCTS expansion loop:
          - If the root agent is terminal and passes evaluation, returns its solution.
          - Otherwise, iteratively selects an agent using UCT, expands it,
            and backpropagates failed terminal children.
          - Finally, if no perfect solution is found, returns the best agent's solution.
        """
        if self.root.is_terminal and self.root.evaluation_passed:
            return f"[MASTER+FastToG] Found solution at root: {self.root.action}"

        for _ in range(self.max_expansion):
            selected_agent = select_with_uct(self.root)
            for _ in range(self.num_branches):
                child_eval_passed, child_solution, child_score = selected_agent.expand()
                if child_eval_passed:
                    return f"[MASTER+FastToG] Found solution: {child_solution}"
                # If the child is terminal but fails evaluation, backpropagate its score.
                child_agent = selected_agent.children[-1]
                if child_agent.is_terminal and not child_agent.evaluation_passed:
                    reward = child_score
                    agent_ptr = selected_agent
                    while agent_ptr is not None:
                        agent_ptr.backpropagate(reward)
                        agent_ptr = agent_ptr.parent

        best_agent = find_best_agent(self.root)
        return f"[MASTER+FastToG] Best attempt: {best_agent.action}"

##############################################
# SECTION E: Example Usage
##############################################
if __name__ == "__main__":
    question = "What kind of clothing should I bring along if I head to the area around the Pennsylvania Convention Center in spring?"
    entity = "Pennsylvania Convention Center"
    max_expansion = 3
    num_branches = 2

    system = MASTERFastToG(question, entity, max_expansion, num_branches)
    final_answer = system.run()
    print(final_answer)
