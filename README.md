# Multi-Arm Robotic System for Large Object Orientation Control

A novel gait-inspired control framework for reorienting very large objects using coordinated multi-arm robotic systems with suction-based end effectors.

## Overview

This repository implements a breakthrough approach to manipulating objects that are significantly larger than the workspace of individual robotic arms. By coordinating multiple UR5e robotic arms in a walking-like pattern, our system can reorient objects up to 5×5×1 meters - approximately five times larger than what any single robot could handle alone.

### Key Features

- **Gait-Inspired Control**: Alternates between "stepping" phases (individual robot repositioning) and "dragging" phases (coordinated object reorientation)
- **Large Object Capability**: Successfully manipulates rectangular objects up to 100° in roll and pitch orientations
- **Multi-Arm Coordination**: Coordinates four UR5e robotic arms with suction-based end effectors
- **Workspace Optimization**: Overcomes individual robot workspace limitations through strategic contact point repositioning
- **Surface-Based Manipulation**: Uses discrete surface representation and optimal path planning for stable manipulation

## System Architecture

The system consists of six main components:

### Core Files

- **`gait_generator.py`** - Implements the gait-inspired control algorithm that creates coordinated stepping and shifting patterns for multi-arm manipulation
- **`path_generator.py`** - Handles surface discretization, contact point selection, and A* path planning for optimal robot trajectories on object surfaces
- **`robot.py`** - Defines the robotic arm model and suction-based end effector capabilities
- **`robot_controller.py`** - Manages individual robot control, inverse kinematics, and suction state coordination
- **`simulation.py`** - PyBullet-based physics simulation environment for testing and validation
- **`run.py`** - Main execution script that orchestrates the complete manipulation pipeline

## Technical Approach

### Algorithm Pipeline

1. **Surface Representation**: Discretizes the object surface into a navigable graph structure
2. **Contact Point Selection**: Identifies optimal suction contact points for initial and target orientations
3. **Path Planning**: Uses A* algorithm to find shortest paths between contact points while avoiding edges
4. **Gait Generation**: Creates coordinated step-and-shift sequences that enable large object reorientation
5. **Command Execution**: Translates high-level plans into specific robot position, orientation, and suction commands

### Key Innovations

- **Exponential Orientation Progression**: Prevents collisions by using smaller steps initially and larger steps as clearance increases
- **Hungarian Algorithm Optimization**: Minimizes total robot travel distance through optimal path assignment
- **Arc-Based Transitions**: Ensures smooth robot movements when stepping between surface contact points

## Experimental Results

The system successfully demonstrates:

- ✅ Roll and pitch rotations up to 100 degrees
- ✅ Manipulation of objects up to 5×5×1 meters
- ✅ Stable control throughout reorientation process
- ✅ Coordination of 4 robotic arms in complex patterns

### Current Limitations

- Limited to rectangular prism objects with flat surfaces
- No yaw rotation capability
- Open-loop control (no real-time feedback)
- Kinematic constraints not fully integrated

## Getting Started

### Prerequisites

- Python 3.8+
- PyBullet physics engine
- NumPy, SciPy for mathematical operations
- Custom URDF models for inverted UR5e configuration

### Running the Simulation

```bash
python run.py
```

This will execute the complete pipeline from path planning through simulation visualization.

## Research Paper & Video

📄 **Paper**: [Orientation Control of Large Objects Using Multi-Arm Robotic Systems with Suction-Based End Effectors](link-to-paper.pdf)

🎥 **Demo Video**: [Watch the multi-arm manipulation in action](https://www.youtube.com/watch?v=Nqc2ppG90kE)

*Author: Daniel Vinals, Department of Mechanical Engineering, Boston University*

## Applications

This technology has potential applications in:

- **Manufacturing**: Handling large panels, sheets, and components
- **Construction**: Positioning building materials and prefabricated elements  
- **Aerospace**: Manipulating large spacecraft components and structures
- **Logistics**: Reorienting oversized packages and cargo
- **Space Robotics**: On-orbit servicing of large satellites and debris

## Future Development

Planned enhancements include:

- Integration of robot kinematic constraints
- Yaw rotation capabilities
- Closed-loop feedback control
- Support for complex object geometries
- Real-world hardware validation

## Citation

If you use this work in your research, please cite:

```bibtex
@article{vinals2024orientation,
  title={Orientation Control of Large Objects Using Multi-Arm Robotic Systems with Suction-Based End Effectors},
  author={Vinals, Daniel},
  journal={Boston University Department of Mechanical Engineering},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to Professor Roberto Tron for guidance and support throughout this research project.
