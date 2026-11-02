"""FrozenLake agentic GRPO with Qwen3-0.6B on TPU (Tunix + vLLM)

Trains **Qwen/Qwen3-0.6B** to play multi-turn **FrozenLake** with agentic GRPO, using the
[Tunix](https://github.com/google/tunix) Python API for training and **vLLM (TPU)** for rollouts.
Everything is defined in this file: the environment, the agent, the dataset and the training setup.

**Recipe** (the configuration that worked best in our runs):
- 100 GRPO steps; 64 prompts × 8 generations per step; up to 2048 prompt and 2048 response tokens
- AdamW (β1 0.9, β2 0.95, no weight decay), peak LR **1e-5**: ramp up over steps 0–20, hold to step 40,
  cosine decay to 0 at step 100
- Loss `gspo-token`, RLOO advantages, no KL penalty (`beta=0`), token-level sampler importance sampling
- Qwen3 **thinking mode on**: the model reasons in a `<think>` block before each move
- Trainer mesh `(fsdp=8, tp=1)`; vLLM rollouts with tensor parallelism 8 on the same chips

**Reference result** (same recipe on a TPU v5e-8): training solve rate went from about 22% to about 70%
(best 10-step window 0.70, best step 0.84) in roughly 5–6 hours.

Requirements: a TPU v5e-8 (8 chips) and Python >= 3.12, with these installed beforehand
(the tested versions): google-tunix[frozenlake] at commit
c4c5da1597bd6a7b3f2ce6ad9ba7e90a34a0c467, vllm-tpu 0.32.0, tpu-inference 0.32.0, jax/jaxlib 0.11.2,
libtpu 0.0.48, flax 0.12.10, optax 0.2.8, orbax-checkpoint 0.12.6, transformers 5.19.0,
grain 0.2.18, gymnasium 1.3.0.

Outputs (train.log, checkpoints) go to $OUTPUT_DIR: /kaggle/working on Kaggle, otherwise
/tmp/frozenlake_qwen3_0p6b.
"""


import sys

assert sys.version_info >= (3, 12), f"Tunix needs Python >= 3.12, got {sys.version.split()[0]}."


# ============================================================================
# 1. Settings, logging and imports
# ============================================================================
# Training logs everything at INFO (including frequent vLLM engine stats). To keep the console output
# small, the full log goes to `$OUTPUT_DIR/train.log`, and only the per-step summaries, weight syncs and
# errors are printed.


import os
import logging as pylogging  # The FrozenLake env code below rebinds `logging` to absl's.

OUTPUT_DIR = os.environ.get(  # train.log and checkpoints
    "OUTPUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/frozenlake_qwen3_0p6b"
)
CKPT_DIR = f"{OUTPUT_DIR}/ckpts"         # trained checkpoints (~6 GB each)
TRAIN_LOG = f"{OUTPUT_DIR}/train.log"    # full training log
TB_LOG_DIR = "/tmp/logging"              # TensorBoard metrics and trajectory logs
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_DOWNLOAD_DIR = "/tmp/models/Qwen3-0.6B"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Must be set before vLLM / JAX are imported.
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"         # route vLLM logs through Python logging (to the file)
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# W&B: online if WANDB_API_KEY is set (in the environment or as a Kaggle secret), otherwise offline.
if "WANDB_API_KEY" not in os.environ:
  try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
  except Exception:
    os.environ["WANDB_MODE"] = "offline"
print("W&B:", os.environ.get("WANDB_MODE", "online"))


class _ProgressFilter(pylogging.Filter):
  # Shows only per-step summaries, weight syncs and errors on the console.
  def filter(self, record):
    msg = record.getMessage()
    return record.levelno >= pylogging.ERROR or "[step " in msg or "Weights synced" in msg


def setup_logging():
  # All INFO logs to TRAIN_LOG; a filtered view to the console.
  from absl import logging as absl_logging
  absl_logging.use_python_logging()
  absl_logging.set_verbosity(absl_logging.INFO)
  root = pylogging.getLogger()
  root.setLevel(pylogging.INFO)
  for h in list(root.handlers):
    root.removeHandler(h)
  fmt = pylogging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
  file_handler = pylogging.FileHandler(TRAIN_LOG)
  file_handler.setFormatter(fmt)
  root.addHandler(file_handler)
  console = pylogging.StreamHandler(sys.stdout)
  console.setFormatter(pylogging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
  console.addFilter(_ProgressFilter())
  root.addHandler(console)


from importlib.metadata import version
import jax
from jax import numpy as jnp
from flax import nnx
import grain
import numpy as np
import optax
import pandas as pd
import datasets as datasets_lib
import transformers
from orbax import checkpoint as ocp

from tunix.models import automodel
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.sft import metrics_logger

setup_logging()
for pkg in ("google-tunix", "vllm-tpu", "tpu-inference", "jax", "libtpu", "flax", "transformers"):
  print(f"{pkg:15s} {version(pkg)}")
print("devices:", jax.devices())
assert jax.device_count() == 8, "This recipe expects a TPU v5e-8 (8 chips)."


# ============================================================================
# 2. FrozenLake environment
# ============================================================================
# A Gymnasium FrozenLake wrapped as a Tunix task environment. Each episode is a random board (2×2 to 8×8)
# with slippery ice. The agent gets reward 1 for reaching the goal within `max_steps` moves.


"""DISCLAIMER:

This implementation is based on the Gymnasium FrozenLake environment and the
RAGEN project:
- Gymnasium: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- RAGEN:
https://github.com/RAGEN-AI/RAGEN/blob/main/ragen/env/frozen_lake/env.py

 Some components have been modified or extended for custom use in this project.
"""

import copy
from typing import Any, Dict
from absl import logging
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from gymnasium.utils import seeding
import numpy as np
from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult

MAX_STEPS: int = 5


# DFS to check that it's a valid path.
def is_valid(board: list[list[str]], max_size: int) -> bool:
  frontier, discovered = [], set()
  # find the start point
  start_r, start_c = np.where(np.array(board) == "S")
  frontier.append((start_r[0], start_c[0], 0))  # row, col steps
  # dfs to check if there is a path from start to goal
  while frontier:
    r, c, steps = frontier.pop()
    if steps > MAX_STEPS:
      continue

    if (r, c) not in discovered:
      discovered.add((r, c))
      directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
      for x, y in directions:
        r_new = r + x
        c_new = c + y
        if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
          continue
        if board[r_new][c_new] == "G":
          return True
        if board[r_new][c_new] != "H":
          frontier.append((r_new, c_new, steps + 1))
  return False


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: int = 0
) -> tuple[list[str], tuple[int, int]]:
  """Generates a random valid map (one that has a path from start to goal)

  Args:
    size: size of each side of the grid
    p: probability that a tile is frozen
    seed: seed to ensure the generation of reproducible maps

  Returns:
    A random valid map
  """
  valid = False
  board: list[list[str]] = []  # initialize to make pyright happy

  np_random, _ = seeding.np_random(seed)

  # generate random start and end points
  goal_r, goal_c = -1, -1
  while not valid:
    p = min(1, p)
    board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p]).tolist()
    while True:
      start_r = int(np_random.integers(0, size))
      start_c = int(np_random.integers(0, size))
      goal_r = int(np_random.integers(0, size))
      goal_c = int(np_random.integers(0, size))
      # Ensure start and goal are different positions
      if (start_r, start_c) != (goal_r, goal_c):
        break
    board[start_r][start_c] = "S"
    board[goal_r][goal_c] = "G"
    valid = is_valid(board, size)
  return ["".join(x) for x in board], (goal_r, goal_c)


def get_goal_position(random_map):
  positions = np.argwhere(random_map == b"G")
  if positions.size == 0:
    return None  # G not found
  return tuple(positions[0])  # returns (row, col)


class FrozenLakeEnv(BaseTaskEnv, GymFrozenLakeEnv):
  """Inherits from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv

  ## Description
  The game starts with the player at random location of the frozen lake grid
  world with the
  goal located at another random location for the 4x4 environment.

  ## Action Space
  The action shape is `(1,)` in the range `{0, 3}` indicating
  which direction to move the player.
  NOTE the action space is different from
  gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
  - 0: Still
  - 1: Left
  - 2: Down
  - 3: Right
  - 4: Up

  ## Starting State
  The episode starts with the player at random location

  ## Rewards
  NOTE added -0.1 as penalty for invalid action
  Reward schedule:
  - Reach goal: +1
  - Reach hole: 0
  - Reach frozen: 0

  ## Arguments
  `is_slippery`: if action is left and is_slippery is True, then:
  - P(move left)=1/3
  - P(move up)=1/3
  - P(move down)=1/3

  ## Example
  P   _   _   _
  _   _   _   O
  O   _   O   _
  O   _   _   G
  """

  # Map gym state in integer
  MAP_LOOKUP = {
      b"P": 0,
      b"F": 1,
      b"H": 2,
      b"G": 3,
  }

  # Define rules to transform to rendered text observation of the environment
  GRID_LOOKUP = {
      0: " P \t",  # player
      1: " _ \t",  # frozen
      2: " O \t",  # hole
      3: " G \t",  # goal
      4: " X \t",  # player fall into hole
      5: " √ \t",  # player on goal
  }

  ACTION_LOOKUP = {
      0: "None",
      1: "Left",
      2: "Down",
      3: "Right",
      4: "Up",
  }

  INVALID_ACTION = 0
  PENALTY_FOR_INVALID = -1

  def __init__(
      self,
      entry: dict[str, Any],
      group_id: int | None = None,
      pair_index: int | None = None,
      max_steps: int = 5,
      **kwargs,
  ):
    global MAX_STEPS
    MAX_STEPS = max_steps

    desc = kwargs.pop("desc", None)
    is_slippery = kwargs.pop("is_slippery", False)
    self.seed = int(entry.get("seed", 42))
    self.size = int(entry.get("size", 8))
    self.p = float(entry.get("p", 0.8))

    if desc is None:
      random_map, goal_position = generate_random_map(
          size=self.size, p=self.p, seed=self.seed
      )
    else:
      random_map = np.asarray(copy.deepcopy(desc), dtype="c")
      goal_position = get_goal_position(random_map)

    self.goal_postion = goal_position

    BaseTaskEnv.__init__(self, max_steps=MAX_STEPS)
    GymFrozenLakeEnv.__init__(self, desc=random_map[:], is_slippery=is_slippery)

    self.ACTION_SPACE = gym.spaces.Discrete(4, start=1)

    self.map_kwargs = {
        "size": self.size,
        "p": self.p,
    }
    self.env_kwargs = {
        "is_slippery": is_slippery,
        "desc": copy.deepcopy(desc),
        "seed": self.seed,
    }
    self.action_map = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
    }

    self.reward = 0
    self._valid_actions = []

    if not hasattr(self, "extra_kwargs"):
      self.extra_kwargs = {}
    self.extra_kwargs["group_id"] = group_id
    self.extra_kwargs["pair_index"] = pair_index

  def _get_player_position(self):
    return (self.s // self.ncol, self.s % self.ncol)

  def _initial_observation(self) -> Any:
    GymFrozenLakeEnv.reset(self, seed=self.seed)
    self.reward = 0
    self._valid_actions = []
    init_observation = self.render(mode="tiny_rgb_array")
    return init_observation

  def finished(self):
    player_pos = self._get_player_position()
    return self.desc[player_pos] in b"GH"

  def success(self):
    """Check if the agent has reacched the goal (G) or hole (H)"""
    player_pos = self._get_player_position()
    return self.desc[player_pos] in b"G"

  def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:  # type: ignore[signature-mismatch]
    return BaseTaskEnv.step(self, action)

  def _step_impl(self, action: Any) -> EnvStepResult:
    """- Map custom action to gymnasium FrozenLakeEnv action and take the step

    - Check if the action is effective (whether player moves in the env).
    """
    if self.success():
      return EnvStepResult(
          observation=self.render(),
          reward=1.0,
          done=True,
          info={"action_is_effective": False},
      )

    if not action:
      action = self.INVALID_ACTION
    action = int(action)

    assert isinstance(action, int), "Action must be an integer"
    assert not self.success(), "Agent has already reached the goal or hole"

    if action == self.INVALID_ACTION:
      return EnvStepResult(
          observation=self.render(),
          reward=0.0,
          done=False,
          info={"action_is_effective": False},
      )

    prev_player_position = int(self.s)

    player_pos, reward, done, _, prob = GymFrozenLakeEnv.step(
        self, int(self.action_map[action])
    )

    obs = self.render()
    return EnvStepResult(
        observation=obs,
        reward=float(reward),
        done=done,
        info={"action_is_effective": prev_player_position != int(player_pos)},
    )

  def render(self, mode="tiny_rgb_array"):
    assert mode in ["tiny_rgb_array", "list", "state", "rgb_array", "ansi"]
    if mode in ["rgb_array", "ansi"]:
      prev_render_mode = self.render_mode
      self.render_mode = mode
      obs = GymFrozenLakeEnv.render(self)
      self.render_mode = prev_render_mode
      return obs
    room_state = copy.deepcopy(self.desc)

    # replace the position of start 'S' with 'F'
    position_S = np.where(room_state == b"S")
    room_state[position_S] = b"F"

    # replace the position of the player with 'P'
    position_P = self._get_player_position()
    room_state[position_P] = b"P"

    if mode == "state":
      # transform 'S', 'F', 'H', 'G' to numpy integer array
      room_state = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room_state)
      # add player in hole or player on goal
      if self.desc[position_P] == b"H":
        room_state[position_P] = 4
      elif self.desc[position_P] == b"G":
        room_state[position_P] = 5
      return room_state

    room_state = self.render(mode="state").tolist()

    if mode == "list":
      lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
      return [" ".join(lookup(cell) for cell in row) for row in room_state]

    if mode == "tiny_rgb_array":
      lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
      result = "\n".join(
          "".join(lookup(cell) for cell in row) for row in room_state
      )
      # result += f"Player Position is at ({position_P[0]}, {position_P[1]}), Goal Position is at ({self.goal_postion[0]}, {self.goal_postion[1]})"
      return result

  def reset(self):  # type: ignore[signature-mismatch]
    BaseTaskEnv.reset(self)
    GymFrozenLakeEnv.reset(self, seed=self.seed)
    return self.render(mode="tiny_rgb_array"), {}

  @classmethod
  def from_dict(cls, env_info: dict) -> "FrozenLakeEnv":
    return cls(
        entry=env_info,
        max_turns=env_info.get("max_turns", MAX_STEPS),
        is_slippery=env_info.get("is_slippery", False),
    )


# ============================================================================
# 3. FrozenLake agent
# ============================================================================
# Turns the board into a chat prompt, and parses the model's reply into an action (`Up`, `Down`, `Left`,
# `Right`) for the environment. It uses the multi-step prompt, which asks the model to reason before acting.


import re
from typing import Any, Optional

from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent

# Prompting format inspired by the RAGEN project: https://github.com/RAGEN-AI/RAGEN
SYSTEM_PROMPT: str = """You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.

Please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""

MULTI_SHOT_SYSTEM_PROMPT: str = """You are a helpful assistant. You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.

Below are examples for an interaction:
Example1:
User: Current Observation:
P   _   _   _   _
O   _   _   O   _
O   _   O   _   _
O   _   _   G   _
_   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now at the top right corner. It should reach G at the bottom right corner. I should move it closer to it. I can move right or down but there is a hole in down position and I can not move diagonally. There is no hole in my next movement right so I can move to right. Action: ```Right```

Example2:
User: Current Observation:
_   _   _   _
_   _   _   O
_   O   _   P
O   _   _   G
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now at the near G. It should reach G to its bottom. I should move to be on it. There is no hole in my next movement so I can move to down. Action: ```Down```

Example3:
User: Current Observation:
_   _   _   O   _
O   _   P   O   _
O   _   O   _   _
O   _   _   G   _
_   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: G is at the bottom right relative to P. I want to move closer so I should move right or down. But there is a hole at each position and I do not want to fall into holes. Up and left are both valid but left brings me closer. Action: ```Left```

Example4:
User: Current Observation:
_   _   _   _
_   _   _   O
_   O   _   O
O   G   P   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now near G. But game has not finished. P is not at G and I should never output invalid action. I need to recheck my understanding. P is not actually on G yet because they are not overlapping, it needs reach G to its left. Action: ```Left```

Example5:
User: Current Observation:
_   _   _   O   _
O   _   P   _   _
O   _   O   O   O
O   _   O   G   _
O   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: G is at the bottom right corner of P. I can move left, right, or up. Move right will initially bring me closer but I can't reach G that way. Move up and left means I can still reach G. Move up will result in 9 steps in total while left is 7 steps. I need to move left. Action: ```Left```

Now it is your turn, please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""


class FrozenLakeAgent(base_agent.ConversationAgentBase):

  def __init__(
      self,
      system_prompt: Optional[str] = None,
      use_multistep_prompt: bool | None = True,
  ):
    self.multistep_prompt = use_multistep_prompt
    system_prompt = (
        SYSTEM_PROMPT
        if not self.multistep_prompt
        else MULTI_SHOT_SYSTEM_PROMPT
    )
    super().__init__(system_prompt=system_prompt)
    self.last_observation = None

  def _init_messages(self, system_prompt: str) -> None:
    """Initialize conversation history with a system prompt.

    Subclasses may override this to inject additional content (e.g., tool
    documentation) into the initial system message.

    Args:
      system_prompt: The system prompt to use.
    """
    self._messages = [{"role": "system", "content": system_prompt or ""}]

  def update_from_env(
    self,
    observation: Any,
    reward: float,
    done: bool,
    info: dict[str, Any] | None = None,
    **kwargs,
  ) -> None:
    new_obs_str = str(observation)
    # Base message for the user
    new_obs_str = "Current Observation: \n" + new_obs_str
    if not done:
      new_obs_str += "\n" + "You have not achieved the goal, P has not reached G yet. Please give the next action."

    # Check if the observation is the same as the previous step's observation
    if self.last_observation and self.last_observation == new_obs_str:
      new_obs_str += "\nYour last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response. Remember, you should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```."
    self.last_observation = new_obs_str

    super().update_from_env(new_obs_str, reward, done, info)
    self.cur_step = agent_types.Step(observation=new_obs_str)

  def _observation_to_messages(
      self, observation: Any, reward: float, done: bool, info: dict[str, Any]
  ) -> None:
    self._messages.append({"role": "user", "content": str(observation)})

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    DIRECTION_MAP = {"left": 1, "down": 2, "right": 3, "up": 4}

    thought = response
    action_str = str(FrozenLakeEnv.INVALID_ACTION)

    matches = re.findall(r"```(.*?)```", response, re.DOTALL)

    if matches:
      last_match_content = matches[-1].strip()
      last_match_index = response.rfind(f"```{last_match_content}```")
      if last_match_index != -1:
        thought = response[:last_match_index].strip()

      extracted_text = last_match_content.lower()

      if extracted_text in DIRECTION_MAP:
        action_str = str(DIRECTION_MAP[extracted_text])
      elif extracted_text.isdigit() and int(extracted_text) in DIRECTION_MAP.values():
        action_str = str(int(extracted_text))

    # Add assistant's response to conversation history.
    self._messages.append({"role": "assistant", "content": response})

    self._trajectory.steps.append(self.cur_step)
    # Record complete step with conversation context and parsed action.
    cur_step = self._trajectory.steps[-1]
    cur_step.thought = thought
    cur_step.action = action_str  # pyrefly: ignore[bad-assignment]
    cur_step.model_response = response

    self.step += 1
    return agent_types.Action(action=cur_step.action)

  def reset(self) -> None:
    super().reset()
    self.last_observation = None


# ============================================================================
# 4. Dataset
# ============================================================================
# 10,000 random board configurations (seed, size, slip probability), seed 42, the same generator as the
# Tunix FrozenLake example. Training uses the first 100 batches of 64 (6,400 boards), in order. The `prompts`
# field is empty because the agent builds each prompt from the environment state.


def get_frozenlake_dict(seed: int, size: int, p: float) -> dict:
  """
  Create a dictionary with FrozenLake environment configuration parameters.

  Args:
      seed: Random seed for environment generation
      size: Grid size (size x size grid)
      p: Probability of moving in the intended direction (1-p = slip probability)

  Returns:
      Dictionary containing environment configuration with keys: seed, size, p
  """
  return {"env_name": "frozenlake", "seed": int(seed), "size": int(size), "p": float(p)}


def generate_dataset_parameters(
    size: int, random_seed: int = 42, grid_size_range: tuple[int, int] = (2, 9)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Generate random parameters for FrozenLake environments.

  This function creates diverse environment configurations by sampling:
  - Random seeds for environment generation
  - Grid sizes ranging from 2x2 to 9x9
  - Slip probabilities between 0.15-0.4 (p values 0.6-0.85)

  Args:
      size: Number of environment configurations to generate
      random_seed: Random seed for reproducible parameter generation
      grid_size_range: Lower (inclusive) and upper (exclusive) grid size bounds

  Returns:
      Tuple of (seeds, sizes, p_values) numpy arrays
  """
  np.random.seed(random_seed)
  seeds = np.random.randint(0, 100000, size=size)
  sizes = np.random.randint(
      *grid_size_range, size=size
  )  # Grid sizes from 2x2 to 9x9
  p_values = np.random.uniform(
      0.6, 0.85, size=size
  )  # Slip probability between 0.15-0.4

  return seeds, sizes, p_values


BATCH_SIZE = 64
NUM_STEPS = 100  # one batch per GRPO step

seeds, sizes, ps = generate_dataset_parameters(10_000, random_seed=42, grid_size_range=(2, 9))
train_df = pd.DataFrame([get_frozenlake_dict(s, sizes[i], ps[i]) for i, s in enumerate(seeds)])


def add_empty_prompt(item):
  item["prompts"] = ""
  return item


train_dataset = (
    grain.MapDataset.source(datasets_lib.Dataset.from_pandas(train_df))
    .map(add_empty_prompt)[: NUM_STEPS * BATCH_SIZE]
    .to_iter_dataset()
    .batch(BATCH_SIZE)
)
print(train_df.head(3))


# ============================================================================
# 5. Models
# ============================================================================
# The reference model is loaded in bfloat16. The actor (the model being trained) is a float32 copy:
# at learning rates around 1e-5, Adam updates fall below bfloat16 precision and would be rounded away.
# Both compute in bfloat16, with activation rematerialization per decoder layer.


mesh = jax.make_mesh((8, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)

reference_model, _ = automodel.AutoModel.from_pretrained(
    MODEL_ID,
    mesh,
    model_source=automodel.ModelSource.HUGGINGFACE,
    model_download_path=MODEL_DOWNLOAD_DIR,
    rng_seed=42,
    remat_config="DECODER",
    dtype="bfloat16",
    use_flash_attention=False,
)
graphdef, params = nnx.split(reference_model)
actor_model = nnx.merge(graphdef, jax.tree.map(lambda x: x.astype(jnp.float32), params))

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
# Qwen3 thinking mode ON (the parser's default): the model reasons in a <think> block before
# each action. The runs this recipe comes from used thinking mode; turning it off gives much
# shorter, lower-entropy responses and a different training trajectory.
chat_parser = chat_parser_lib.QwenChatTemplateParser(tokenizer, enable_thinking=True)


# ============================================================================
# 6. Optimizer and learning-rate schedule
# ============================================================================
# `sgdr_schedule` chains two warmup-cosine segments. The first warms up to 1e-5 over 20 steps, then "decays"
# from 1e-5 to 1e-5 (a hold) until step 40. The second is the real cosine decay to 0 at step 100.
# No gradient clipping: the runs this recipe comes from had none, and gradient norms stayed around 0.003.


lr_schedule = optax.schedules.sgdr_schedule([
    dict(init_value=1e-7, peak_value=1e-5, warmup_steps=20, decay_steps=40, end_value=1e-5),
    dict(init_value=1e-5, peak_value=1e-5, warmup_steps=0, decay_steps=60, end_value=0.0),
])
optimizer = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=0.0)
print({step: f"{float(lr_schedule(step)):.2e}" for step in (0, 10, 20, 40, 70, 99)})


# ============================================================================
# 7. Rollout (vLLM) and RL cluster
# ============================================================================
# The trainer and vLLM share the 8 chips. vLLM starts with random weights and gets the actor's weights synced
# in before the first rollout and after every training step. Checkpoints (model + optimizer state) are saved
# at steps 50 and 100, keeping only the latest, to `$OUTPUT_DIR/ckpts/actor/<step>`.


MAX_PROMPT_LENGTH = 2048
MAX_RESPONSE_LENGTH = 2048

rollout_config = base_rollout.RolloutConfig(
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_tokens_to_generate=MAX_RESPONSE_LENGTH,
    kv_cache_size=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 256,
    temperature=0.7,
    top_p=1.0,
    top_k=0,  # no top-k / top-p filtering, so sampler and trainer logprobs are comparable
    return_logprobs=True,
    rollout_vllm_model_version=MODEL_ID,
    rollout_vllm_init_with_random_weights=True,
    rollout_vllm_sampling_kwargs={"skip_special_tokens": False},
    rollout_vllm_hbm_utilization=0.2,
    rollout_vllm_tpu_backend_type="jax",
    rollout_vllm_server_mode=True,
    rollout_vllm_server_mode_submission_threshold=0,
    rollout_vllm_server_mode_submission_timeout_s=0.0,
    rollout_vllm_async_scheduling=False,
    tensor_parallel_size=8,
    data_parallel_size=1,
    rollout_vllm_max_num_seqs=32,
    rollout_vllm_max_num_batched_tokens=8192,
    rollout_vllm_kwargs={
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
        "dtype": "bfloat16",
    },
)

training_config = rl_engine_lib.RLTrainingConfig(
    actor_optimizer=optimizer,
    max_steps=NUM_STEPS,
    mini_batch_size=64,
    train_micro_batch_size=1,
    compute_logps_chunk_size=1024,
    eval_every_n_steps=10,
    metrics_logging_options=metrics_logger.MetricsLoggerOptions(
        log_dir=TB_LOG_DIR, project_name="tunix-frozenlake", flush_every_n_steps=1
    ),
    checkpoint_root_directory=CKPT_DIR,
    checkpointing_options=ocp.CheckpointManagerOptions(save_interval_steps=50, max_to_keep=1),
)

cluster_config = rl_engine_lib.ClusterConfig(
    role_to_mesh={
        rl_engine_lib.Role.ACTOR: mesh,
        rl_engine_lib.Role.REFERENCE: mesh,
        rl_engine_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine="vllm",
    offload_to_cpu=False,
    training_config=training_config,
    rollout_config=rollout_config,
)

rl_engine = rl_engine_lib.RLEngine(
    actor=actor_model,
    reference=reference_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)


# ============================================================================
# 8. Train
# ============================================================================
# Agentic GRPO: for each of the 64 boards in a batch, the agent plays 8 episodes (up to 8 moves each) against
# the environment, and the policy is updated from the episodes' rewards. Expect the first step to take
# longer (compilation), then roughly 2–5 minutes per step.


grpo_config = GRPOConfig(
    num_generations=8,
    num_iterations=1,
    max_response_length=MAX_RESPONSE_LENGTH,
    beta=0.0,
    epsilon=0.003,
    epsilon_high=0.005,
    loss_algo="gspo-token",
    loss_agg_mode="sequence-mean-token-mean",
    kl_loss_mode="low_var_kl",
    advantage_estimator="rloo",
    sampler_is="token",
    sampler_is_threshold=2.0,
    max_concurrency=512,
)

learner = GRPOLearner(
    rl_engine=rl_engine,
    algo_config=grpo_config,
    chat_parser=chat_parser,
    agent_class=FrozenLakeAgent,
    agent_kwargs={"use_multistep_prompt": True},
    env_class=FrozenLakeEnv,
    env_kwargs={"max_steps": 8},
)

setup_logging()  # re-apply: libraries imported since may have added their own handlers
learner.train(train_dataset)


# ============================================================================
# 9. Results
# ============================================================================
# The GRPO training reward is the fraction of episodes that reached the goal (the solve rate).


import re
rows = []
for line in open(TRAIN_LOG):
  m = re.search(r"\[step (\d+)\] train_reward=([0-9.]+).*?compl_len=([0-9.]+).*?entropy=([0-9.]+)", line)
  t = re.search(r"time=([0-9.]+)s", line)
  if m:
    rows.append((int(m[1]), float(m[2]), float(m[3]), float(m[4]), float(t[1]) if t else 0.0))

print(f"{'steps':>9}  {'reward':>7}  {'min':>6}  {'max':>6}  {'compl_len':>9}  {'entropy':>7}  {'step_time':>9}")
for i in range(0, len(rows), 10):
  w = rows[i:i + 10]
  r = [x[1] for x in w]
  print(f"{w[0][0]:>4}-{w[-1][0]:<4}  {sum(r)/len(r):7.3f}  {min(r):6.3f}  {max(r):6.3f}  "
        f"{sum(x[2] for x in w)/len(w):9.0f}  {sum(x[3] for x in w)/len(w):7.3f}  {sum(x[4] for x in w)/len(w):8.0f}s")
if rows:
  print(f"\n{len(rows)} steps; best step reward {max(x[1] for x in rows):.3f}; "
        f"training time {sum(x[4] for x in rows) / 3600:.1f} h")


# Saved checkpoint(s): model params + optimizer state.
for step in sorted(os.listdir(f"{CKPT_DIR}/actor")) if os.path.isdir(f"{CKPT_DIR}/actor") else []:
  print(step, sorted(os.listdir(f"{CKPT_DIR}/actor/{step}")))
