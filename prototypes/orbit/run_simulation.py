# Core
import sys
import time
import os

# Manipulation + Draw + CLI
import numpy as np
import pygame
import pygame.gfxdraw
import fire
from tqdm import trange, tqdm

# Dev
from pprint import pprint as pp

# Default colors
COLOR_BLACK = (0, 0, 0)

# Pygame Constants
TIME_DELTA = 0.001
TIME_EVENT_ID = pygame.USEREVENT+1
WIDTH = 1620
HEIGHT = 1080

# Simulation "constants"
NUM_DIMS = 2
G = 39.478  # 6.67408e-11
NUM_OF_BODIES = 6
FLOAT_DTYPE = np.float
SUN_INDEX = 0
MIN_POSITION, MAX_POSITION = (10, 100)

# Visualization constants
HIST_TIMESTEPS = 100
TIMESTEP_DELAY = 2
DATA_FOLDER = './data/{}'.format(NUM_OF_BODIES)


def generate_initial_values():
    """
    Generate the new physical values used in the simulation
    """
    v = np.zeros((NUM_OF_BODIES, NUM_DIMS), dtype=FLOAT_DTYPE)
    v2 = np.zeros((NUM_OF_BODIES, NUM_DIMS), dtype=FLOAT_DTYPE)
    p2 = np.random.uniform(low=MIN_POSITION,
                           high=MAX_POSITION, size=(NUM_OF_BODIES, NUM_DIMS))
    p2[:, 1] = np.random.uniform(
        low=0, high=2*np.pi, size=NUM_OF_BODIES).astype(FLOAT_DTYPE)
    p = np.zeros((NUM_OF_BODIES, NUM_DIMS), dtype=FLOAT_DTYPE)
    p[:, 0] = p2[:, 0] * np.cos(p2[:, 1])
    p[:, 1] = p2[:, 0] * np.sin(p2[:, 1])

    m = np.random.uniform(0.02, 9, size=(
        NUM_OF_BODIES, 1)).astype(FLOAT_DTYPE)
    f = np.zeros((NUM_OF_BODIES, NUM_DIMS), dtype=FLOAT_DTYPE)
    d = np.zeros((NUM_OF_BODIES, NUM_OF_BODIES,
                  NUM_DIMS), dtype=FLOAT_DTYPE)

    # Configure color
    c = np.random.randint(64, 255, size=(NUM_OF_BODIES, 3))

    r = np.log2(2 * m)

    return v, v2, p, p2, m, f, d, c, r


def compute_orbit(v, p, m, r, c, orbit_type):
    """
    Make the velocities to adjust to a given orbit_type
    """

    # Assert orbit_type is valid
    if orbit_type not in ["random", "circular", "elliptical"]:
        raise NotImplementedError(
            "No such orbit type: \"{}\"".format(orbit_type))

    # Random calculation is easy
    if orbit_type == "random":
        v = np.random.uniform(
            low=-3, high=3, size=(NUM_OF_BODIES, NUM_DIMS)).astype(FLOAT_DTYPE)
        return v, p, m, r, c

    # Orbit is circular or elliptical, so the only change is
    # a multiplication in the end of the below code

    # Set up the sun at the center, with 100 mass and a yellow color
    p[SUN_INDEX, :] = [0, 0]
    m[SUN_INDEX, :] = [100]
    r[SUN_INDEX, :] = 2 * np.log2(np.sum(r[SUN_INDEX + 1:, :]))
    c[SUN_INDEX, :] = [255, 255, 0]

    center_of_mass = np.sum(p*m, axis=0)/np.sum(m)
    distance_to_center = np.linalg.norm(p - center_of_mass, axis=1)
    _, _, centripetal_force = nbody(TIME_DELTA, p, v, m, G=G)
    for i in range(NUM_OF_BODIES):
        u_force = centripetal_force[i] / \
            np.linalg.norm(centripetal_force[i])
        velocity_magnitude = np.sqrt(np.linalg.norm(centripetal_force[i])
                                     * np.linalg.norm(distance_to_center[i]) / m[i])

        rotation = 1 if np.random.choice(
            ["clockwise", "counterclockwise"]) == "clockwise" else -1
        v[i, 0] = u_force[1] * rotation
        v[i, 1] = -u_force[0] * rotation
        v[i] *= velocity_magnitude * (np.interp(np.linalg.norm(p), [MIN_POSITION, MAX_POSITION], [
                                      1.5, 2/3]) if orbit_type == "elliptical" else 1)

    return v, p, m, r, c


def nbody(dt, pos, vel, mass, radii=None, out_pos=None, out_vel=None,
          force_placeholder=None, distance_placeholder=None, G=39.478, epsilon=1e-3):
    """
    Compute the physical interaction between n-bodies
    """
    out_pos, out_vel, force_placeholder = map(lambda x: np.empty_like(
        pos) if x is None else x, (out_pos, out_vel, force_placeholder))

    n = pos.shape[0]

    if distance_placeholder is None:
        distance_placeholder = np.zeros((n, n, 2), dtype=pos.dtype)
    d1 = pos.view()[np.newaxis, :, :]
    d2 = pos.view()[:, np.newaxis, :]
    distance_placeholder = d2 - d1

    force_placeholder[:] = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                force_placeholder[i] -= distance_placeholder[i, j, :] * \
                    (G * mass[i] * mass[j] /
                     (epsilon + np.linalg.norm(distance_placeholder[i, j, :]) ** 2))

    out_vel = np.add(vel, force_placeholder / mass * dt, out=out_vel)
    out_pos = np.add(pos, (vel + (out_vel - vel) / 2) * dt, out=out_pos)
    return out_pos, out_vel, force_placeholder


def run_simulation(draw:bool=False, save_data:bool=False, start_at:int=0, num_scenes:int=1000, max_timesteps:int=1000, num_of_bodies:int=6, orbit_type:str="elliptical"):
    """
    Run the simulation for `num_scenes` with `num_timesteps` each scene.
    Properly resets the environment and the physical values each scene.
    """

    global NUM_OF_BODIES, DATA_FOLDER
    NUM_OF_BODIES = num_of_bodies
    DATA_FOLDER = './data/{}'.format(NUM_OF_BODIES)

    # Run num_scenes simulations
    for curr_scene in trange(start_at, start_at + int(num_scenes), desc="Scene"):
        # Create save_data folder
        if save_data:
            os.makedirs("{}/{}".format(DATA_FOLDER, curr_scene), exist_ok=True)

        # Generate values
        v, v2, p, p2, m, f, d, c, r = generate_initial_values()

        # Compute the orbit
        v, p, m, r, c = compute_orbit(v, p, m, r, c, orbit_type)

        # Configure history
        hp = np.empty((HIST_TIMESTEPS, NUM_OF_BODIES,
                       NUM_DIMS), dtype=np.float)
        hp[:] = np.nan

        # Configure pyGame canvas
        if draw:
            pygame.init()
            # screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            screen = pygame.display.set_mode((WIDTH, HEIGHT))

            pygame.time.set_timer(TIME_EVENT_ID, int(1000*TIME_DELTA))

            # Configure variables to draw
            delay = 0
            radius = 0
            # Umax, Umin, passed_iter = float("-inf"), float("inf"), 0

        # Generate placeholders
        all_p, all_v, all_m, all_r, all_f, all_data = np.zeros(
            [max_timesteps, *p.shape]), np.zeros(
            [max_timesteps, *v.shape]), np.zeros(
            [max_timesteps, *m.shape]), np.zeros(
            [max_timesteps, *r.shape]), np.zeros(
            [max_timesteps, *f.shape]), np.zeros(
            [max_timesteps, 1])

        # Run this scene for max_timesteps steps
        for curr_timestep in trange(max_timesteps, desc="Timestep"):
            if draw:
                redraw = False
                while not redraw:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            sys.exit()
                        if event.type == TIME_EVENT_ID:
                            if save_data:
                                for all_x, x in zip([all_p, all_v, all_m, all_r, all_f, all_data], [p, v, m, r, f, np.array([G])]):
                                    all_x[curr_timestep] = x.copy()

                            nbody(
                                TIME_DELTA,
                                p,
                                v,
                                m,
                                out_pos=p2,
                                out_vel=v2,
                                force_placeholder=f,
                                distance_placeholder=d,
                                G=G
                            )

                            # Swap position and velocities
                            p, p2 = p2, p
                            v, v2 = v2, v

                            # End the loop, to be able to redraw the simulation
                            redraw = True
                        # end if
                    # end for
                # end while

                # Update the timesteps
                delay = (delay + 1) % TIMESTEP_DELAY
                if not delay:
                    for t in range(HIST_TIMESTEPS-1):
                        hp[t, :, :] = hp[t+1, :, :]
                hp[HIST_TIMESTEPS-1, :, :] = p[:, :]

                # Recompute the max_p and radius, used to configure zoom
                avg_p = np.mean(p, axis=0)
                if orbit_type == "circular" or orbit_type == "elliptical":
                    avg_p = p[0]
                max_p = np.max(np.linalg.norm(
                    p-avg_p[np.newaxis, :], axis=1))
                radius = max(1.5 * max_p, radius)
                # radius = 1.5*max_p # Uncomment for trippier visualisation

                if HEIGHT < WIDTH:
                    h = 2 * radius
                    w = h * WIDTH/HEIGHT
                else:
                    w = 2 * radius
                    h = w * HEIGHT/WIDTH

                # Redraw the planets/sun in the simulation
                screen.fill(COLOR_BLACK)
                for t in range(HIST_TIMESTEPS):
                    for i in range(NUM_OF_BODIES):
                        if not (np.isnan(hp[t, i])).any():
                            pygame.gfxdraw.filled_circle(
                                screen,
                                int(WIDTH/2 +
                                    (hp[t, i, 0] - avg_p[0]) * WIDTH/w),
                                int(HEIGHT/2 + (hp[t, i, 1] -
                                                avg_p[1]) * HEIGHT/h),
                                int(r[i, 0] * min(WIDTH, HEIGHT) / radius),
                                list(c[i]) + [255 // (HIST_TIMESTEPS-t)]
                            )

                # Flip color buffer
                pygame.display.flip()
            else:
                if save_data:
                    for all_x, x in zip([all_p, all_v, all_m, all_r, all_f, all_data], [p, v, m, r, f, np.array([G])]):
                        all_x[curr_timestep] = x.copy()

                nbody(
                    TIME_DELTA,
                    p,
                    v,
                    m,
                    out_pos=p2,
                    out_vel=v2,
                    force_placeholder=f,
                    distance_placeholder=d,
                    G=G
                )

                # Swap position and velocities
                p, p2 = p2, p
                v, v2 = v2, v
            # end if
        # end for

        # Save values
        if save_data:
            np.save("{}/{}/pos.npy".format(DATA_FOLDER, str(curr_scene)), all_p)
            np.save("{}/{}/vel.npy".format(DATA_FOLDER, str(curr_scene)), all_v)
            np.save("{}/{}/mass.npy".format(
                DATA_FOLDER, str(curr_scene)), all_m)
            np.save("{}/{}/radii.npy".format(
                DATA_FOLDER, str(curr_scene)), all_r)
            np.save("{}/{}/force.npy".format(
                DATA_FOLDER, str(curr_scene)), all_f)
            np.save("{}/{}/data.npy".format(
                DATA_FOLDER, str(curr_scene)), all_data)
    # end for


if __name__ == "__main__":
    fire.Fire(run_simulation)
