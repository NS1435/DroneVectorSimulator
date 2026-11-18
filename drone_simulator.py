import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

# Define the simulation field, drone speed, and timing.
FIELD_SIZE = 15.0
START_POSITION = (7.5, 7.5)
DRONE_SPEED = 2.0
TIME_STEP = 0.05
FRAME_INTERVAL = 50
MAX_TRAIL_POINTS = 2000
CROSSWIND_TOLERANCE = 5.0

# Prevent Matplotlib from using arrow keys for navigation.
plt.rcParams["keymap.back"] = []
plt.rcParams["keymap.forward"] = []


# --------------------------------------------------
# VECTOR MATH
# --------------------------------------------------

# Contains the vector operations used by the simulator.
class VectorMath:

    @staticmethod
    def magnitude(vector):
        # Calculate vector length using the Pythagorean theorem.
        return float(np.linalg.norm(vector))

    @staticmethod
    def normalize(vector):
        # Divide a vector by its magnitude to get a unit vector.
        vector = np.asarray(vector, dtype=float)
        length = VectorMath.magnitude(vector)

        # The zero vector has no direction.
        if length == 0:
            return None

        return vector / length

    @staticmethod
    def dot_product(a, b):
        # Dot product: a · b = ax*bx + ay*by
        return float(np.dot(a, b))

    @staticmethod
    def projection(a, b):
        # Find the component of vector a along vector b.
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        # Projection = [(a · b) / (b · b)] * b
        denominator = VectorMath.dot_product(b, b)

        if denominator == 0:
            return None

        return VectorMath.dot_product(a, b) / denominator * b

    @staticmethod
    def angle(a, b):
        # Use the dot product to calculate the angle between vectors.
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        mag_a = VectorMath.magnitude(a)
        mag_b = VectorMath.magnitude(b)

        if mag_a == 0 or mag_b == 0:
            return None

        # cos(theta) = (a · b) / (|a| * |b|)
        cosine = VectorMath.dot_product(a, b) / (mag_a * mag_b)

        # Prevent rounding errors from exceeding the valid range.
        cosine = np.clip(cosine, -1, 1)

        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def resultant(a, b):
        # Add two vectors component by component.
        return (
            np.asarray(a, dtype=float)
            + np.asarray(b, dtype=float)
        )

    @staticmethod
    def subtract(a, b):
        # Subtract two vectors component by component.
        return (
            np.asarray(a, dtype=float)
            - np.asarray(b, dtype=float)
        )

    @staticmethod
    def scalar_multiply(scalar, vector):
        # Multiply each vector component by a scalar.
        return scalar * np.asarray(vector, dtype=float)

    @staticmethod
    def components(vector, direction):
        # Separate a vector into parallel and perpendicular parts.
        parallel = VectorMath.projection(vector, direction)

        if parallel is None:
            return None, None

        perpendicular = VectorMath.subtract(
            vector, parallel
        )

        return parallel, perpendicular


# --------------------------------------------------
# DRONE
# --------------------------------------------------

# Stores the drone's position, velocity, and flight history.
class Drone:

    def __init__(self):
        self._start_position = None
        self._position = None
        self._velocity = None
        self._actual_velocity = None
        self._distance_traveled = None
        self._simulation_time = None
        self._flight_path = None

        self.reset()

    def reset(self):
        # Return the drone to its original position.
        self._start_position = np.array(
            START_POSITION, dtype=float
        )

        self._position = self._start_position.copy()

        self._velocity = np.zeros(2)
        self._actual_velocity = np.zeros(2)

        self._distance_traveled = 0.0
        self._simulation_time = 0.0

        # Store up to 2000 positions for the flight trail.
        self._flight_path = deque(maxlen=MAX_TRAIL_POINTS)
        self._flight_path.append(self._position.copy())

    def set_velocity(self, velocity):
        self._velocity = np.asarray(
            velocity, dtype=float
        ).copy()

    def set_actual_velocity(self, velocity):
        self._actual_velocity = np.asarray(
            velocity, dtype=float
        ).copy()

    def move(self, wind, dt):
        # Add the drone's velocity and wind velocity.
        # Vr = Vd + Vw
        intended_velocity = VectorMath.resultant(
            self._velocity, wind
        )

        old_position = self._position.copy()

        # Update position using:
        # r_new = r_old + Vr * dt
        new_position = VectorMath.resultant(
            old_position,
            VectorMath.scalar_multiply(
                dt, intended_velocity
            )
        )

        # Keep the drone inside the flight boundary.
        self._position = np.clip(
            new_position, 0, FIELD_SIZE
        )

        # Find the displacement during this time step.
        displacement = VectorMath.subtract(
            self._position, old_position
        )

        # Calculate actual velocity after boundary restrictions.
        self._actual_velocity = VectorMath.scalar_multiply(
            1.0 / dt, displacement
        )

        # Accumulate total distance traveled.
        distance = VectorMath.magnitude(displacement)
        self._distance_traveled += distance

        self._simulation_time += dt

        # Only add a trail point when the drone moves.
        if distance > 0:
            self._flight_path.append(self._position.copy())

    def get_position(self):
        return self._position.copy()

    def get_start_position(self):
        return self._start_position.copy()

    def get_velocity(self):
        return self._velocity.copy()

    def get_actual_velocity(self):
        return self._actual_velocity.copy()

    def get_displacement(self):
        # Displacement = current position - starting position.
        return VectorMath.subtract(
            self._position, self._start_position
        )

    def get_distance_traveled(self):
        return self._distance_traveled

    def get_simulation_time(self):
        return self._simulation_time

    def get_flight_path(self):
        return list(self._flight_path)


# --------------------------------------------------
# DRONE SIMULATOR
# --------------------------------------------------

# Handles the graphical interface and user controls.
class DroneSimulator:

    def __init__(self):
        self.drone = Drone()

        self.running = False
        self.pressed_keys = set()

        self.setup_graph()
        self.setup_controls()

        # Connect keyboard events to their corresponding methods.
        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.handle_key_press
        )

        self.fig.canvas.mpl_connect(
            "key_release_event",
            self.handle_key_release
        )

        self.fig.canvas.mpl_connect(
            "figure_leave_event",
            self.clear_keys
        )

        # Call update() repeatedly to animate the drone.
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            interval=FRAME_INTERVAL,
            cache_frame_data=False
        )

    # ----------------------------------------------
    # GRAPH
    # ----------------------------------------------

    def setup_graph(self):
        self.fig = plt.figure(figsize=(15, 9))

        # Place the flight field on the left side.
        self.ax = self.fig.add_axes(
            [0.07, 0.24, 0.52, 0.68]
        )

        self.ax.set_xlim(0, FIELD_SIZE)
        self.ax.set_ylim(0, FIELD_SIZE)
        self.ax.set_aspect("equal")

        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")

        self.ax.set_title(
            "Interactive Drone Vector Simulator — MATH 215"
        )

        self.ax.grid(True, alpha=0.3)

        # Draw the boundary of the flight area.
        self.boundary = Rectangle(
            (0, 0),
            FIELD_SIZE,
            FIELD_SIZE,
            fill=False,
            linestyle=":",
            edgecolor="gray",
            linewidth=2,
            label="Flight Boundary"
        )

        self.ax.add_patch(self.boundary)

        start = self.drone.get_start_position()

        # Mark the starting position.
        self.ax.plot(
            start[0],
            start[1],
            "kx",
            markersize=12,
            label="Starting Position"
        )

        # Create the drone marker and flight path.
        self.drone_marker, = self.ax.plot(
            [],
            [],
            "ko",
            markersize=10,
            label="Drone"
        )

        self.flight_line, = self.ax.plot(
            [],
            [],
            color="green",
            linewidth=2,
            label="Flight Path"
        )

        # Dashed line shows displacement from the starting point.
        self.displacement_line, = self.ax.plot(
            [],
            [],
            "k--",
            linewidth=2,
            label="Displacement"
        )

        # Create three arrows representing the velocity vectors.
        self.drone_arrow = self.ax.quiver(
            [start[0]],
            [start[1]],
            [0],
            [0],
            color="blue",
            angles="xy",
            scale_units="xy",
            scale=1,
            label="Drone Velocity"
        )

        self.wind_arrow = self.ax.quiver(
            [start[0]],
            [start[1]],
            [0],
            [0],
            color="red",
            angles="xy",
            scale_units="xy",
            scale=1,
            label="Wind Velocity"
        )

        self.resultant_arrow = self.ax.quiver(
            [start[0]],
            [start[1]],
            [0],
            [0],
            color="purple",
            angles="xy",
            scale_units="xy",
            scale=1,
            label="Resultant Velocity"
        )

        # Display the mathematical calculations on the right.
        self.math_text = self.fig.text(
            0.65,
            0.90,
            "",
            fontsize=8,
            family="monospace",
            verticalalignment="top"
        )

        self.ax.legend(
            loc="upper left",
            fontsize=8
        )

    # ----------------------------------------------
    # CONTROLS
    # ----------------------------------------------

    def setup_controls(self):
        # Create sliders for the X and Y wind components.
        wind_x_axis = self.fig.add_axes(
            [0.15, 0.15, 0.42, 0.025]
        )

        wind_y_axis = self.fig.add_axes(
            [0.15, 0.10, 0.42, 0.025]
        )

        self.wind_x_slider = Slider(
            wind_x_axis,
            "Wind X",
            -3,
            3,
            valinit=0
        )

        self.wind_y_slider = Slider(
            wind_y_axis,
            "Wind Y",
            -3,
            3,
            valinit=0
        )

        # Create Start/Pause and Reset buttons.
        start_axis = self.fig.add_axes(
            [0.15, 0.025, 0.18, 0.05]
        )

        reset_axis = self.fig.add_axes(
            [0.39, 0.025, 0.18, 0.05]
        )

        self.start_button = Button(
            start_axis,
            "Start / Pause"
        )

        self.reset_button = Button(
            reset_axis,
            "Reset"
        )

        # Connect buttons to their functions.
        self.start_button.on_clicked(self.toggle_running)
        self.reset_button.on_clicked(self.reset_simulation)

    def get_wind(self):
        # Read the wind vector from the sliders.
        return np.array([
            self.wind_x_slider.val,
            self.wind_y_slider.val
        ])

    def toggle_running(self, event):
        # Switch between running and paused.
        self.running = not self.running

        if not self.running:
            self.drone.set_actual_velocity([0, 0])

        self.update_math_display()
        self.fig.canvas.draw_idle()

    def reset_simulation(self, event):
        # Reset the drone and wind controls.
        self.running = False
        self.pressed_keys.clear()

        self.drone.reset()

        self.wind_x_slider.reset()
        self.wind_y_slider.reset()

        self.update_graphics()
        self.update_math_display()

        self.fig.canvas.draw_idle()

    # ----------------------------------------------
    # KEYBOARD
    # ----------------------------------------------

    def clear_keys(self, event=None):
        self.pressed_keys.clear()
        self.drone.set_velocity([0, 0])

    def handle_key_press(self, event):
        # Add pressed arrow keys to the set.
        if event.key in ["up", "down", "left", "right"]:
            self.pressed_keys.add(event.key)
            self.update_commanded_velocity()

    def handle_key_release(self, event):
        # Remove released arrow keys from the set.
        if event.key in self.pressed_keys:
            self.pressed_keys.remove(event.key)
            self.update_commanded_velocity()

    def update_commanded_velocity(self):
        # Determine direction from the pressed arrow keys.
        direction = np.zeros(2)

        if "up" in self.pressed_keys:
            direction[1] += 1

        if "down" in self.pressed_keys:
            direction[1] -= 1

        if "right" in self.pressed_keys:
            direction[0] += 1

        if "left" in self.pressed_keys:
            direction[0] -= 1

        # Normalize direction so diagonal movement
        # has the same speed as horizontal or vertical movement.
        unit = VectorMath.normalize(direction)

        if unit is None:
            self.drone.set_velocity([0, 0])
        else:
            self.drone.set_velocity(
                VectorMath.scalar_multiply(DRONE_SPEED, unit)
            )

    # ----------------------------------------------
    # ANIMATION
    # ----------------------------------------------

    def update(self, frame):
        # Update the drone's position while running.
        if self.running:
            wind = self.get_wind()

            self.drone.move(
                wind,
                TIME_STEP
            )

        # Refresh the graph and mathematical information.
        self.update_graphics()
        self.update_math_display()

        return []

    # ----------------------------------------------
    # GRAPHICS
    # ----------------------------------------------

    def update_graphics(self):
        position = self.drone.get_position()
        start = self.drone.get_start_position()

        # Update the drone's position on the graph.
        self.drone_marker.set_data(
            [position[0]],
            [position[1]]
        )

        # Draw the recorded flight path.
        path = self.drone.get_flight_path()

        if path:
            points = np.asarray(path)

            self.flight_line.set_data(
                points[:, 0],
                points[:, 1]
            )

        # Draw the displacement from start to current position.
        self.displacement_line.set_data(
            [start[0], position[0]],
            [start[1], position[1]]
        )

        drone_velocity = self.drone.get_velocity()
        wind_velocity = self.get_wind()

        resultant = VectorMath.resultant(
            drone_velocity,
            wind_velocity
        )

        vectors = [
            (self.drone_arrow, drone_velocity),
            (self.wind_arrow, wind_velocity),
            (self.resultant_arrow, resultant)
        ]

        # Update the three velocity arrows.
        for arrow, vector in vectors:
            arrow.set_offsets([position])
            arrow.set_UVC(
                [vector[0]],
                [vector[1]]
            )

    # ----------------------------------------------
    # MATHEMATICS DISPLAY
    # ----------------------------------------------

    def update_math_display(self):
        position = self.drone.get_position()
        start = self.drone.get_start_position()

        drone_velocity = self.drone.get_velocity()
        wind_velocity = self.get_wind()

        # Add drone velocity and wind velocity.
        resultant = VectorMath.resultant(
            drone_velocity,
            wind_velocity
        )

        actual_velocity = self.drone.get_actual_velocity()

        # Calculate speed and unit direction.
        speed = VectorMath.magnitude(resultant)
        unit = VectorMath.normalize(resultant)

        # Calculate displacement from the starting position.
        displacement = self.drone.get_displacement()

        displacement_magnitude = VectorMath.magnitude(
            displacement
        )

        # Calculate dot product and angle between velocities.
        dot = VectorMath.dot_product(
            drone_velocity,
            wind_velocity
        )

        angle = VectorMath.angle(
            drone_velocity,
            wind_velocity
        )

        # Separate wind into parallel and perpendicular components.
        parallel, perpendicular = VectorMath.components(
            wind_velocity,
            drone_velocity
        )

        # Determine whether wind is a headwind, tailwind, or crosswind.
        if angle is None:
            wind_type = "N/A"
            angle_text = "Undefined"
        else:
            angle_text = f"{angle:.2f} degrees"

            if abs(angle - 90) < CROSSWIND_TOLERANCE:
                wind_type = "Crosswind"

            elif angle < 90:
                wind_type = "Tailwind"

            else:
                wind_type = "Headwind"

        # Format vectors for the text display.
        def fmt(vector):
            if vector is None:
                return "Undefined"

            return (
                f"<{vector[0]:.2f}, "
                f"{vector[1]:.2f}>"
            )

        # Align the labels and values.
        def row(label, value):
            return f"{label:<19}{value}"

        status = "RUNNING" if self.running else "PAUSED"

        # Organize the mathematical information by Unit 1 topics.
        math_display = "\n".join([
            "VECTOR MATHEMATICS — UNIT 1",
            "--------------------------------",
            row("Status:", status),
            "",
            "1.1 — VECTORS AND COMPONENTS",
            row("Position r:", fmt(position)),
            row("Start r0:", fmt(start)),
            row("Speed |Vr|:", f"{speed:.2f}"),
            row("Unit Direction:", fmt(unit)),
            "",
            "1.2 — VECTOR OPERATIONS",
            row("Drone Vd:", fmt(drone_velocity)),
            row("Wind Vw:", fmt(wind_velocity)),
            row("Vr = Vd + Vw:", fmt(resultant)),
            row("Actual Velocity:", fmt(actual_velocity)),
            row("d = r - r0:", fmt(displacement)),
            row(
                "Displacement |d|:",
                f"{displacement_magnitude:.2f}"
            ),
            row(
                "Distance:",
                f"{self.drone.get_distance_traveled():.2f}"
            ),
            "",
            "1.3 — EQUATIONS OF LINES",
            "r(t) = r0 + Vr*t",
            "x(t) = x0 + Vrx*t",
            "y(t) = y0 + Vry*t",
            "r0 = start of a constant-velocity",
            "segment, Vr = resultant velocity.",
            "Valid until a boundary is reached.",
            "",
            "1.4 — DOT PRODUCTS AND ANGLES",
            row("Vd . Vw:", f"{dot:.2f}"),
            row("Angle:", angle_text),
            row("Wind Type:", wind_type),
            "",
            "1.5 — VECTOR PROJECTIONS",
            row("Parallel Wind:", fmt(parallel)),
            row("Perpendicular:", fmt(perpendicular)),
            "",
            row(
                "Simulation Time:",
                f"{self.drone.get_simulation_time():.2f} s"
            ),
            "",
            "CONTROLS",
            "Arrow Keys: Move Drone",
            "Wind Sliders: Adjust Wind",
            "Start/Pause: Toggle Motion",
            "Reset: Restart"
        ])

        self.math_text.set_text(math_display)

    # ----------------------------------------------
    # RUN
    # ----------------------------------------------

    def run(self):
        # Initialize the display and open the simulator window.
        self.update_graphics()
        self.update_math_display()

        plt.show()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

# Run the simulator only when this file is executed directly.
if __name__ == "__main__":
    simulator = DroneSimulator()
    simulator.run()