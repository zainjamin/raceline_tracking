import numpy as np
from numpy.typing import NDArray

from racetrack import RaceTrack


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0.0
        self.prev_e = 0.0

    def update(self, measure: float, target: float) -> float:
        error = target - measure

        p_term = self.kp * error

        self.integral += error * self.dt
        i_term = self.ki * self.integral

        derivative = (error - self.prev_e) / self.dt
        self.prev_e = error
        d_term = self.kd * derivative

        return p_term + i_term + d_term


velocity_controller = PIDController(20.0, 0.1, 0.0, 0.1)
steer_controller = PIDController(10.0, 0.0, 0.3, 0.1)


def lower_controller(
    state: NDArray, desired: NDArray, parameters: NDArray
) -> np.ndarray:
    # [steer angle, velocity]
    assert desired.shape == (2,)

    steering_rate = steer_controller.update(state[2], desired[0])
    acceleration = velocity_controller.update(state[3], desired[1])

    return np.array([steering_rate, acceleration])


def line_circle_intersection(a: NDArray, b: NDArray, p: NDArray, d: float):
    # Intersection points is going to be x.
    # |x - p| = d and a + t(b - a) = x
    # x.x - 2x.p + p.p = d.d
    # (a+t(b-a)).(a+t(b-a)) - 2(a+t(b-a)).p + p.p = d.d
    # a.a + 2 * t(a.(b-a)) + (b-a).(b-a) t^2 - 2a.p - 2t(b-a).p + p.p = d.d
    # |b-a|t^2 + (2*a.(b-a) - 2(b-a).p)t + (a.a - 2a.p + p.p - d.d) = 0
    a2 = np.dot(b - a, b - a)
    a1 = np.dot(2 * a, b - a) - 2 * np.dot(b - a, p)
    a0 = np.dot(a, a) - 2 * np.dot(a, p) + np.dot(p, p) - d * d

    disc = a1 * a1 - 4 * a2 * a0

    if disc >= 0:
        sqrt_disc = np.sqrt(disc)
        t1 = (-a1 + sqrt_disc) / (2 * a2)
        t2 = (-a1 - sqrt_disc) / (2 * a2)

        x1 = a + t1 * (b - a) if 0 <= t1 <= 1 else None
        x2 = a + t2 * (b - a) if 0 <= t2 <= 1 else None

        if x1 is None and x2 is None:
            return None

        if x1 is None:
            return x2

        if x2 is None:
            return x1

        # return point closest to b
        return x1 if np.linalg.norm(b - x1) < np.linalg.norm(b - x2) else x2

    return None


last_idx = 0


def pure_pusuit(state: NDArray, path: NDArray, parameters: NDArray) -> None:
    global last_idx
    start_idx = last_idx

    x, y, _, v, theta = state

    cur_pos = np.array([x, y])
    goal = path[last_idx]

    min_lookahead = 1.0
    max_lookahead = 20.0
    k_lookahead = 0.6
    lookahead_distance = min(max_lookahead, max(min_lookahead, k_lookahead * v))
    for i in range(start_idx, len(path) - 1):
        a = path[i]
        b = path[i + 1]
        intersection = line_circle_intersection(a, b, cur_pos, lookahead_distance)

        goal = intersection
        if goal is not None:
            last_idx = i
            if np.linalg.norm(intersection - b) < np.linalg.norm(cur_pos - b):
                break
        else:
            goal = path[last_idx]

    alpha = np.arctan2(goal[1] - y, goal[0] - x) - theta
    steering_angle = np.arctan2(2.0 * parameters[0] * np.sin(alpha), lookahead_distance)
    return steering_angle


v_profile = None


def generate_speed_profile(path: NDArray, parameters: NDArray):
    n = len(path)

    deltas = np.diff(path, axis=0, prepend=path[0].reshape(1, -1))
    dists = np.linalg.norm(deltas, axis=1)

    curvatures = np.zeros(n)
    for i in range(1, n - 1):
        p1 = path[i - 1]
        p2 = path[i]
        p3 = path[i + 1]

        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a > 0 and b > 0 and c > 0:
            s = (a + b + c) / 2.0
            area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
            if area > 1e-6:
                radius = (a * b * c) / (4.0 * area)
                curvatures[i] = 1.0 / radius
            else:
                curvatures[i] = 0.0

    v_profile = np.zeros(n)
    for i in range(n):
        k = max(curvatures[i], 1e-6)
        allowed_v = np.sqrt(parameters[10] / k)
        v_profile[i] = min(allowed_v, parameters[5])

    v_profile[-1] = 0.0
    for i in range(n - 2, -1, -1):
        dist = dists[i + 1]
        max_reachable_v = np.sqrt(v_profile[i + 1] ** 2 - 2 * parameters[8] * dist)
        v_profile[i] = min(v_profile[i], max_reachable_v)

    return v_profile


def controller(
    state: NDArray,
    parameters: NDArray,
    racetrack: RaceTrack,
) -> NDArray:

    global v_profile, last_idx

    if v_profile is None:
        v_profile = generate_speed_profile(racetrack.target_path, parameters)

    steering_angle = pure_pusuit(state, racetrack.target_path, parameters)
    velocity = v_profile[last_idx]

    return np.array([steering_angle, velocity])
