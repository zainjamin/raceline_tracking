import numpy as np
from numpy.typing import NDArray

from racecar import RaceCar
from racetrack import RaceTrack

controller_state = {
    "vref": 0.0,
    "deltaref": 0.0,
    "int_e_v": 0.0,
    "int_e_d": 0.0,
    "dt": 0.1,
}


def compute_curvature(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p2 - p0)
    if a * b * c < 1e-6:
        return 0.0

    area = (
        abs(p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]))
        * 0.5
    )
    return (4.0 * area) / (a * b * c)


def compute_signed_curvature(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p2 - p0)
    if a * b * c < 1e-6:
        return 0.0

    v1 = p1 - p0
    v2 = p2 - p1
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    area = (
        abs(p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]))
        * 0.5
    )

    sign = 1.0 if cross_product >= 0 else -1.0
    return sign * (4.0 * area) / (a * b * c)


def lower_controller(
    state: NDArray, desired: NDArray, parameters: NDArray
) -> np.ndarray:
    # [steer angle, velocity]
    assert desired.shape == (2,)

    dt = controller_state.get("dt", 0.001)

    e_v = desired[1] - state[3]
    e_d = desired[0] - state[2]
    e_d = np.arctan2(np.sin(e_d), np.cos(e_d))

    controller_state["vref"] = desired[1]
    controller_state["deltaref"] = desired[0]

    controller_state["int_e_v"] += dt * e_v
    controller_state["int_e_d"] += dt * e_d
    controller_state["int_e_v"] = np.clip(controller_state["int_e_v"], -20.0, 20.0)
    controller_state["int_e_d"] = np.clip(controller_state["int_e_d"], -5.0, 5.0)

    kpv = 10.0
    kiv = 0.1

    kpd = 10.0
    kid = 0.1

    acceleration = kpv * e_v + kiv * controller_state["int_e_v"]
    steering_rate = kpd * e_d + kid * controller_state["int_e_d"]

    acceleration = np.clip(
        acceleration,
        parameters[8],
        parameters[10],
    )
    steering_rate = np.clip(
        steering_rate,
        parameters[7],
        parameters[9],
    )

    return np.array([steering_rate, acceleration])


def controller(
    state: NDArray,
    parameters: NDArray,
    racetrack: RaceTrack,
) -> NDArray:
    # The car state advances with a fixed simulation step (racecar.time_step = 0.1),
    # so use that here instead of wall-clock time to avoid over-rate-limiting when
    # the GUI runs slowly.
    dt = 0.1
    controller_state["dt"] = dt

    car_pos = np.array([state[0], state[1]])

    def rate_limit(target: float, previous: float, max_rate: float, dt: float) -> float:
        """
        Rate limits the target value to the previous value +/- the max rate * dt.
        """
        delta = np.clip(target - previous, -max_rate * dt, max_rate * dt)
        return previous + delta

    def wrap_angle(angle: float) -> float:
        """
        Wraps the angle to the range [-pi, pi].
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    centerline = racetrack.centerline
    closest_idx = int(np.argmin(np.linalg.norm(centerline - car_pos, axis=1)))

    i0 = closest_idx
    i1 = (closest_idx + 1) % len(centerline)
    i2 = (closest_idx + 2) % len(centerline)
    local_curvature = compute_curvature(centerline[i0], centerline[i1], centerline[i2])
    local_curvature_signed = compute_signed_curvature(
        centerline[i0], centerline[i1], centerline[i2]
    )

    lookahead_dist = np.clip(7.0 + 0.7 * state[3], 7.0, 28.0)
    if local_curvature > 0.05:
        lookahead_dist = max(2.5, lookahead_dist - 6.0)
    elif local_curvature > 0.02:
        lookahead_dist = max(4.0, lookahead_dist - 4.0)
    elif local_curvature > 0.01:
        lookahead_dist = max(5.5, lookahead_dist - 2.5)
    elif local_curvature > 0.005:
        lookahead_dist = max(6.5, lookahead_dist - 1.0)
    else:
        lookahead_dist = max(7.5, lookahead_dist - 0.5)

    def find_lookahead_point(start_idx: int, distance: float) -> np.ndarray:
        remaining = distance
        idx = start_idx
        while remaining > 0:
            next_idx = (idx + 1) % len(centerline)
            seg = centerline[next_idx] - centerline[idx]
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-6:
                return centerline[next_idx]
            if remaining <= seg_len:
                return centerline[idx] + seg * (remaining / seg_len)
            remaining -= seg_len
            idx = next_idx
        return centerline[idx]

    target_point = find_lookahead_point(closest_idx, lookahead_dist)

    dx = target_point[0] - state[0]
    dy = target_point[1] - state[1]
    angle_to_target = np.arctan2(dy, dx)
    alpha = wrap_angle(angle_to_target - state[4])
    steering_angle = np.arctan2(2.0 * parameters[0] * np.sin(alpha), lookahead_dist)
    steering_angle = wrap_angle(steering_angle)

    max_curv_ahead = local_curvature
    distance_limit = max(60.0, min(140.0, 80.0 + 0.8 * state[3]))
    curve_trigger = 0.02
    tight_curve_trigger = 0.03
    distance_to_curve = float("inf")
    traveled = 0.0
    offset = 1
    max_curv_within_brake = local_curvature

    max_curv_positive = (
        max(0.0, local_curvature_signed) if local_curvature_signed > 0 else 0.0
    )
    max_curv_negative = (
        max(0.0, -local_curvature_signed) if local_curvature_signed < 0 else 0.0
    )

    curves_ahead = []

    while traveled < distance_limit and offset < len(centerline) - 2:
        j0 = (closest_idx + offset - 1) % len(centerline)
        j1 = (closest_idx + offset) % len(centerline)
        j2 = (closest_idx + offset + 1) % len(centerline)
        seg_len = np.linalg.norm(centerline[j1] - centerline[j0])
        traveled += seg_len
        curv = compute_curvature(centerline[j0], centerline[j1], centerline[j2])
        curv_signed = compute_signed_curvature(
            centerline[j0], centerline[j1], centerline[j2]
        )

        max_curv_ahead = max(max_curv_ahead, curv)
        max_curv_within_brake = max(max_curv_within_brake, curv)

        if curv_signed > 0:
            max_curv_positive = max(max_curv_positive, curv_signed)
        elif curv_signed < 0:
            max_curv_negative = max(max_curv_negative, -curv_signed)

        if curv > curve_trigger:
            curves_ahead.append((traveled, curv, curv_signed))
            if distance_to_curve == float("inf"):
                distance_to_curve = traveled

        offset += 1

    def curve_speed_from_curvature(curv: float) -> float:
        if curv < 0.010:
            return 80.0
        elif curv < 0.04:
            return 60.0
        elif curv < 0.08:
            return 45.0
        elif curv < 0.10:
            return 35.0
        else:
            return 22.0

    base_curv_speed = curve_speed_from_curvature(max_curv_ahead)

    if len(curves_ahead) > 1:
        curve_speeds = [
            curve_speed_from_curvature(curv_unsigned)
            for _, curv_unsigned, _ in curves_ahead
        ]
        min_required_speed = min(curve_speeds)

        first_curve = curves_ahead[0]
        second_curve = curves_ahead[1]
        distance_between = second_curve[0] - first_curve[0]
        first_sign = first_curve[2]
        second_sign = second_curve[2]
        is_s_curve = (
            first_sign * second_sign < 0
            and abs(first_sign) > 0.01
            and abs(second_sign) > 0.01
        )

        if is_s_curve and distance_between < 40.0:
            min_required_speed *= 0.7
        elif is_s_curve and distance_between < 60.0:
            min_required_speed *= 0.85

        if distance_between < 50.0:
            min_required_speed *= 0.9

        base_curv_speed = min_required_speed

    straight_speed = 0.0
    if max_curv_ahead < 8e-4 and abs(alpha) < 0.06:
        straight_speed = 75.0
    elif max_curv_ahead < 0.002:
        straight_speed = 60.0

    worst_curv = max(local_curvature, 0.8 * max_curv_ahead + 0.2 * local_curvature)
    curve_speed = min(base_curv_speed, curve_speed_from_curvature(worst_curv))
    if max_curv_within_brake > tight_curve_trigger:
        curve_speed = min(
            curve_speed, curve_speed_from_curvature(max_curv_within_brake)
        )

    high_speed = max(curve_speed, straight_speed)

    max_decel = 12.0
    if distance_to_curve == float("inf") or state[3] <= curve_speed:
        blend = 1.0
    else:
        braking_need = (state[3] ** 2 - curve_speed**2) / (2.0 * max_decel + 1e-6)
        buffer = 6.0
        start_brake = max(5.0, braking_need - buffer)
        end_brake = braking_need + buffer
        if distance_to_curve > end_brake:
            blend = 1.0
        elif distance_to_curve < start_brake:
            blend = 0.0
        else:
            blend = (distance_to_curve - start_brake) / (end_brake - start_brake + 1e-6)
            blend = np.clip(blend, 0.0, 1.0)

    target_velocity = curve_speed + blend * (high_speed - curve_speed)

    if len(curves_ahead) > 1:
        pass

    target_velocity = min(target_velocity, base_curv_speed)

    closest_pt = centerline[closest_idx]
    xte = np.linalg.norm(car_pos - closest_pt)

    if xte > 0.5:
        penalty_factor = 1.0 / (1.0 + 0.5 * (xte - 0.5))
        target_velocity *= penalty_factor

    if abs(alpha) > 0.15:
        penalty_factor = 1.0 / (1.0 + 2.0 * (abs(alpha) - 0.15))
        target_velocity *= penalty_factor

    target_velocity = np.maximum(target_velocity, 10.0)

    velocity = rate_limit(
        target_velocity,
        controller_state["vref"],
        parameters[10],
        dt,
    )

    steering_angle = rate_limit(
        steering_angle,
        controller_state["deltaref"],
        parameters[9],
        dt,
    )
    steering_angle = np.clip(
        steering_angle,
        parameters[1],
        parameters[4],
    )

    return np.array([steering_angle, velocity])
