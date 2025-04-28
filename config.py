from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math
import numpy as np

@dataclass
class SimulationConfig:
    GRAVITY: tuple = (0, 0, -9.81)
    DT: float = 0.01
    NUM_STEPS: int = 500000
    WORKSPACE_RADIUS: float = 3

@dataclass
class RobotConfig:
    URDF_PATH: str = "urdf/ur5e_sc_down.urdf"
    CONTROL_JOINTS: list = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    MAX_FORCE: float = 5000
    MAX_VELOCITY: float = 10
    MAX_EE_VELOCITY: float = 30
    MAX_EE_AVELOCITY: float = 30



@dataclass
class RobotTeam:
    ROBOT_RADIUS: float = 0.5
    NUM_ROBOTS: int = 4
    CENTER: tuple= (0, 0, 1.6)

@dataclass
class Item:
    X_Y_CENTER: tuple = (0, 0)
    HEIGHT_FROM_ROBOT: float = 0.75
    MIN_DIMENSIONS: tuple = (0.5, 0.5, 0.5)
    MAX_DIMENSIONS: tuple = (1.5, 1.5, 1.5)
    MASS: float = 1
    COLOR: tuple = (1, 0, 0, 1)
