import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

FIELD_SIZE = 15.0
START_POSITION = (7.5, 7.5)
DRONE_SPEED = 2.0
TIME_STEP = 0.05
FRAME_INTERVAL = 50
MAX_TRAIL_POINTS = 2000
CROSSWIND_TOLERANCE = 5.0

plt.rcParams["keymap.back"] = []
plt.rcParams["keymap.forward"] = []


# --------------------------------------------------
# VECTOR MATH
# --------------------------------------------------

class VectorMath:

    @staticmethod
    def magnitude(vector):
        return float(np.linalg.norm(vector))

    @staticmethod
    def normalize(vector):
        vector = np.asarray(vector, dtype=float)
        length = VectorMath.magnitude(vector)

        if length == 0:
            return None

        return vector / length

    @staticmethod
    def dot_product(a, b):
        return float(np.dot(a, b))

    @staticmethod
    def projection(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        denominator = VectorMath.dot_product(b, b)

        if denominator == 0:
            return None

        return VectorMath.dot_product(a, b) / denominator * b

    @staticmethod
    def angle(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        mag_a = VectorMath.magnitude(a)
        mag_b = VectorMath.magnitude(b)

        if mag_a == 0 or mag_b == 0:
            return None

        cosine = VectorMath.dot_product(a, b) / (mag_a * mag_b)
        cosine = np.clip(cosine, -1, 1)

        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def resultant(a, b):
        return (
            np.asarray(a, dtype=float)
            + np.asarray(b, dtype=float)
        )

    @staticmethod
    def subtract(a, b):
        return (
            np.asarray(a, dtype=float)
            - np.asarray(b, dtype=float)
        )

    @staticmethod
    def scalar_multiply(scalar, vector):
        return scalar * np.asarray(vector, dtype=float)

    @staticmethod
    def components(vector, direction):
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
        self._start_position = np.array(
            START_POSITION, dtype=float
        )

        self._position = self._start_position.copy()

        self._velocity = np.zeros(2)
        self._actual_velocity = np.zeros(2)

        self._distance_traveled = 0.0
        self._simulation_time = 0.0

        self._flight_path = deque(
            maxlen=MAX_TRAIL_POINTS
        )

        self._flight_path.append(
            self._position.copy()
        )

    def set_velocity(self, velocity):
        self._velocity = np.asarray(
            velocity, dtype=float
        ).copy()

    def set_actual_velocity(self, velocity):
        self._actual_velocity = np.asarray(
            velocity, dtype=float
        ).copy()

    def move(self, wind, dt):
        intended_velocity = VectorMath.resultant(
            self._velocity, wind
        )

        old_position = self._position.copy()

        # r_new = r_old + Vr * dt
        new_position = VectorMath.resultant(
            old_position,
            VectorMath.scalar_multiply(
                dt, intended_velocity
            )
        )

        self._position = np.clip(
            new_position, 0, FIELD_SIZE
        )

        displacement = VectorMath.subtract(
            self._position, old_position
        )

        # Actual velocity = displacement / dt
        self._actual_velocity = VectorMath.scalar_multiply(
            1.0 / dt, displacement
        )

        distance = VectorMath.magnitude(
            displacement
        )

        self._distance_traveled += distance
        self._simulation_time += dt

        # Only record a new point if the drone moved.
        if distance > 0:
            self._flight_path.append(
                self._position.copy()
            )

    def get_position(self):
        return self._position.copy()

    def get_start_position(self):
        return self._start_position.copy()

    def get_velocity(self):
        return self._velocity.copy()

    def get_actual_velocity(self):
        return self._actual_velocity.copy()

    def get_displacement(self):
        return VectorMath.subtract(
            self._position,
            self._start_position
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

class DroneSimulator:

    def __init__(self):
        self.drone = Drone()

        self.running = False
        self.pressed_keys = set()

        self.setup_graph()
        self.setup_controls()

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
        self.fig = plt.figure(
            figsize=(15, 9)
        )

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

        self.ax.grid(
            True, alpha=0.3
        )

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

        self.ax.add_patch(
            self.boundary
        )

        start = self.drone.get_start_position()

        self.ax.plot(
            start[0],
            start[1],
            "kx",
            markersize=12,
            label="Starting Position"
        )

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

        self.displacement_line, = self.ax.plot(
            [],
            [],
            "k--",
            linewidth=2,
            label="Displacement"
        )

        # Create the velocity arrows once.
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

        # Compact mathematics panel.
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

        self.start_button.on_clicked(
            self.toggle_running
        )

        self.reset_button.on_clicked(
            self.reset_simulation
        )

    def get_wind(self):
        return np.array([
            self.wind_x_slider.val,
            self.wind_y_slider.val
        ])

    def toggle_running(self, event):
        self.running = not self.running

        if not self.running:
            self.drone.set_actual_velocity(
                [0, 0]
            )

        self.update_math_display()
        self.fig.canvas.draw_idle()

    def reset_simulation(self, event):
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

        self.drone.set_velocity(
            [0, 0]
        )

    def handle_key_press(self, event):
        if event.key in [
            "up", "down", "left", "right"
        ]:
            self.pressed_keys.add(
                event.key
            )

            self.update_commanded_velocity()

    def handle_key_release(self, event):
        if event.key in self.pressed_keys:
            self.pressed_keys.remove(
                event.key
            )

            self.update_commanded_velocity()

    def update_commanded_velocity(self):
        direction = np.zeros(2)

        if "up" in self.pressed_keys:
            direction[1] += 1

        if "down" in self.pressed_keys:
            direction[1] -= 1

        if "right" in self.pressed_keys:
            direction[0] += 1

        if "left" in self.pressed_keys:
            direction[0] -= 1

        unit = VectorMath.normalize(
            direction
        )

        if unit is None:
            self.drone.set_velocity(
                [0, 0]
            )
        else:
            self.drone.set_velocity(
                VectorMath.scalar_multiply(
                    DRONE_SPEED,
                    unit
                )
            )

    # ----------------------------------------------
    # ANIMATION
    # ----------------------------------------------

    def update(self, frame):
        if self.running:
            wind = self.get_wind()

            self.drone.move(
                wind,
                TIME_STEP
            )

        self.update_graphics()
        self.update_math_display()

        return []

    # ----------------------------------------------
    # GRAPHICS
    # ----------------------------------------------

    def update_graphics(self):
        position = self.drone.get_position()
        start = self.drone.get_start_position()

        self.drone_marker.set_data(
            [position[0]],
            [position[1]]
        )

        path = self.drone.get_flight_path()

        if path:
            points = np.asarray(path)

            self.flight_line.set_data(
                points[:, 0],
                points[:, 1]
            )

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

        for arrow, vector in vectors:
            arrow.set_offsets(
                [position]
            )

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

        resultant = VectorMath.resultant(
            drone_velocity,
            wind_velocity
        )

        actual_velocity = self.drone.get_actual_velocity()

        speed = VectorMath.magnitude(
            resultant
        )

        unit = VectorMath.normalize(
            resultant
        )

        displacement = self.drone.get_displacement()

        displacement_magnitude = VectorMath.magnitude(
            displacement
        )

        dot = VectorMath.dot_product(
            drone_velocity,
            wind_velocity
        )

        angle = VectorMath.angle(
            drone_velocity,
            wind_velocity
        )

        parallel, perpendicular = VectorMath.components(
            wind_velocity,
            drone_velocity
        )

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

        def fmt(vector):
            if vector is None:
                return "Undefined"

            return (
                f"<{vector[0]:.2f}, "
                f"{vector[1]:.2f}>"
            )

        def row(label, value):
            return f"{label:<19}{value}"

        status = (
            "RUNNING"
            if self.running
            else "PAUSED"
        )

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

        self.math_text.set_text(
            math_display
        )

    # ----------------------------------------------
    # RUN
    # ----------------------------------------------

    def run(self):
        self.update_graphics()
        self.update_math_display()

        plt.show()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    simulator = DroneSimulator()
    simulator.run()