# Drone Vector Simulator

An interactive 2D drone simulator built with Python to visualize vector mathematics, velocity, displacement, and the effects of wind on movement.

## Overview

Drone Vector Simulator is a Python application that demonstrates how vectors can be used to model motion in a two-dimensional environment.

Users can control a drone using their keyboard, adjust wind conditions in real time, and observe how different velocity vectors combine to influence the drone's movement.

The application provides a graphical representation of the drone's flight path alongside live mathematical calculations, making it a useful tool for exploring vector mathematics, physics, and basic navigation principles.

## Features

- **Interactive Movement:** Control the drone using directional arrow keys.
- **Wind Simulation:** Adjust horizontal and vertical wind velocities using sliders.
- **Real-Time Visualization:** Display drone velocity, wind velocity, and resultant velocity as vectors.
- **Flight Path Tracking:** Visualize the drone's movement over time.
- **Mathematical Analysis:** Display live calculations of vector magnitudes, angles, dot products, and projections.
- **Displacement Tracking:** Compare total distance traveled with displacement from the starting position.
- **Simulation Controls:** Start, pause, and reset the simulation.

## Mathematical Concepts

### Vector Operations

Vector addition is used to calculate the drone's resultant velocity:

$$
\vec{v}_{resultant} = \vec{v}_{drone} + \vec{v}_{wind}
$$

Vector subtraction and scalar multiplication are used throughout the simulation to calculate changes in position and velocity.

### Position and Motion

The drone's position is updated using its velocity and elapsed time:

$$
\vec{r}_{new} = \vec{r}_{current} + \vec{v}_{resultant}\Delta t
$$

### Dot Products and Angles

Dot products are used to calculate the angle between the drone's commanded velocity and wind velocity:

$$
\vec{a}\cdot\vec{b} = |\vec{a}||\vec{b}|\cos(\theta)
$$

### Vector Projections

Vector projections are used to determine how much of the wind acts parallel or perpendicular to the drone's direction of movement:

$$
\mathrm{proj}_{\vec{b}}(\vec{a}) =
\frac{\vec{a}\cdot\vec{b}}{\vec{b}\cdot\vec{b}}\vec{b}
$$

These calculations help illustrate headwinds, tailwinds, and crosswinds.

## Technologies

- **Python 3** - Core programming language
- **NumPy** - Vector calculations and numerical operations
- **Matplotlib** - Interactive visualization, animation, and user controls

The application uses object-oriented programming to separate mathematical calculations, drone behavior, and graphical visualization.

## Installation

### Prerequisites

- Python 3
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DroneVectorSimulator.git
cd DroneVectorSimulator
```

### 2. Install Dependencies

```bash
python3 -m pip install numpy matplotlib
```

### 3. Run the Simulator

```bash
python3 drone_simulator.py
```

## Controls

| Input | Action |
|---|---|
| ↑ | Move north |
| ↓ | Move south |
| ← | Move west |
| → | Move east |
| Wind X slider | Adjust horizontal wind |
| Wind Y slider | Adjust vertical wind |
| Start/Pause | Toggle simulation |
| Reset | Reset drone position and flight data |

### Vector Visualization

| Color | Representation |
|---|---|
| Blue | Commanded drone velocity |
| Red | Wind velocity |
| Purple | Resultant velocity |
| Green | Flight path |
| Black dashed line | Displacement |

## Testing

The repository includes a separate mathematical verification program containing 12 tests.

These tests verify:

- Vector magnitude and normalization
- Vector addition and subtraction
- Scalar multiplication
- Dot products
- Angles between vectors
- Vector projections
- Parallel and perpendicular components
- Handling of undefined vector operations

Run the tests using:

```bash
python3 test_vector_math.py
```

## Project Structure

```text
DroneVectorSimulator/
├── drone_simulator.py     # Main application
├── test_vector_math.py    # Mathematical verification tests
├── diagram.png            # Program architecture diagram
└── README.md              # Documentation
```

## Limitations

This simulator uses a simplified two-dimensional model of drone movement.

It does not currently model real-world flight dynamics such as acceleration, inertia, gravity, motor thrust, or aerodynamic forces.

Wind is represented as an additional velocity vector rather than a physical force acting on the drone.

The simulation is intended for mathematical visualization and experimentation rather than accurate flight prediction.

## Future Development

Potential improvements include:

- Three-dimensional movement and visualization
- Acceleration and more realistic flight dynamics
- Sensor simulation and position estimation
- Autonomous navigation and waypoint tracking
- Integration with microcontrollers and physical hardware

## Author

*Nate Stubbs*

Electrical Engineering
