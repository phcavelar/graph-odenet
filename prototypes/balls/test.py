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
COLOR_GREY = (192, 192, 192)

# Visualization constants
HIST_TIMESTEPS = 5
TIMESTEP_DELAY = 2
DATA_FOLDER = './prototypes/balls/data/'

# Pygame Constants
TIME_DELTA = 0.01
TIME_EVENT_ID = pygame.USEREVENT+1
WIDTH = 800
HEIGHT = 600
DRAW_SCALE = 50

# Simulation "constants"
NUM_DIMS = 2
NUM_OF_BODIES = 9
WALL_GAP = np.random.uniform(
    0.333 * NUM_OF_BODIES, 1 * NUM_OF_BODIES, size=(2,))
WALL_SIZE = (NUM_OF_BODIES + 0.1 - WALL_GAP) / 2
RECTANGLES = [
    pygame.Rect(0, 0, WALL_SIZE[0] * DRAW_SCALE,
                (WALL_SIZE[1] * 2 + WALL_GAP[1]) * DRAW_SCALE),
    pygame.Rect(WALL_SIZE[0] * DRAW_SCALE, (WALL_SIZE[1] + WALL_GAP[1]) * DRAW_SCALE + 0.5, WALL_GAP[0] * DRAW_SCALE + 1,
                WALL_SIZE[1] * DRAW_SCALE + 0.5),
    pygame.Rect((WALL_SIZE[0] + WALL_GAP[0]) * DRAW_SCALE, 0, WALL_SIZE[0] * DRAW_SCALE,
                (WALL_SIZE[1] * 2 + WALL_GAP[1]) * DRAW_SCALE),
    pygame.Rect(WALL_SIZE[0] * DRAW_SCALE, 0, (WALL_SIZE[0] + WALL_GAP[0]) * DRAW_SCALE,
                WALL_SIZE[1] * DRAW_SCALE)
]


def reset_environment():
    """
    Reset the WALL_GAP, WALL_SIZE and RECTANGLES
    variables, to a new random value
    """
    global WALL_GAP, WALL_SIZE, RECTANGLES

    WALL_GAP = np.random.uniform(
        0.333 * NUM_OF_BODIES, 1 * NUM_OF_BODIES, size=(2,))
    WALL_SIZE = (NUM_OF_BODIES + 0.1 - WALL_GAP) / 2

    RECTANGLES = [
        pygame.Rect(0, 0, WALL_SIZE[0] * DRAW_SCALE,
                    (WALL_SIZE[1] * 2 + WALL_GAP[1]) * DRAW_SCALE),
        pygame.Rect(WALL_SIZE[0] * DRAW_SCALE, (WALL_SIZE[1] + WALL_GAP[1]) * DRAW_SCALE + 0.5, WALL_GAP[0] * DRAW_SCALE + 1,
                    WALL_SIZE[1] * DRAW_SCALE + 0.5),
        pygame.Rect((WALL_SIZE[0] + WALL_GAP[0]) * DRAW_SCALE, 0, WALL_SIZE[0] * DRAW_SCALE,
                    (WALL_SIZE[1] * 2 + WALL_GAP[1]) * DRAW_SCALE),
        pygame.Rect(WALL_SIZE[0] * DRAW_SCALE, 0, (WALL_SIZE[0] + WALL_GAP[0]) * DRAW_SCALE,
                    WALL_SIZE[1] * DRAW_SCALE)
    ]


def generate_initial_values():
    """
    Generate the new physical values used in the simulation
    """
    v = np.random.uniform(-5, 5, size=(NUM_OF_BODIES, NUM_DIMS))
    v2 = np.copy(v)
    p = np.empty((NUM_OF_BODIES, NUM_DIMS))
    for i in range(NUM_DIMS):
        p[:, i] = np.random.uniform(
            WALL_SIZE[i], WALL_SIZE[i] + WALL_GAP[i], size=(NUM_OF_BODIES, ))
    p2 = np.copy(p)

    r = np.random.uniform(0.1, 0.3, size=(NUM_OF_BODIES, 1))
    m = np.random.uniform(0.75, 1.25, size=(NUM_OF_BODIES, 1))
    d = np.zeros((NUM_OF_BODIES, NUM_OF_BODIES), dtype=np.float)
    collision = np.zeros(
        (NUM_OF_BODIES, NUM_OF_BODIES), dtype=np.float)

    # Configure color
    c = np.empty((NUM_OF_BODIES, 3))
    c[:, 0] = np.interp(m, [0.75, 1.25], [0, 255]).squeeze()
    c[:, 1] = 255 - np.interp(m, [0.75, 1.25], [0, 255]).squeeze()
    c[:, 2] = 255 - np.interp(m, [0.75, 1.25], [0, 255]).squeeze()

    return v, v2, p, p2, r, m, d, c, collision


def interaction(dt, pos, vel, mass, radii=None, collision=None, out_pos=None, out_vel=None, distance=None, save_data=False):
    """
    Compute the physical interaction between n-many balls
    """
    collision, out_pos, out_vel = map(lambda x: np.empty_like(
        pos) if x is None else x, (collision, out_pos, out_vel))

    n = pos.shape[0]

    if distance is None:
        distance = np.zeros((n, n), dtype=pos.dtype)
    d1 = pos.view()[np.newaxis, :, :]
    d2 = pos.view()[:, np.newaxis, :]
    distance = np.linalg.norm(d2 - d1, axis=2)

    if radii is None:
        radii = np.log2(mass ** 2)

    # Compute collisions in the upper triangle matrix
    collision[:] = False
    for i in range(n):
        for j in range(i + 1, n):
            collision[i, j] = (distance[i, j] <= radii[i] + radii[j])
            if collision[i, j] and i != j:
                direction = (pos[i] - pos[j]) / np.linalg.norm(pos[i] - pos[j])
                direction_sized = direction * \
                    (radii[i] + radii[j] - distance[i, j])
                pos[i] += direction_sized
                pos[j] -= direction_sized

    out_vel[:] = vel.copy()
    for i in range(n):
        for j in range(i + 1, n):
            if collision[i, j]:  # If they are colliding
                out_vel[i] -= (2 * mass[j] /
                               (mass[i] + mass[j])) * (np.dot(vel[i] - vel[j], pos[i] - pos[j]) / (np.linalg.norm(pos[i] - pos[j]) ** 2)) * (pos[i] - pos[j])
                out_vel[j] -= (2 * mass[i] /
                               (mass[j] + mass[i])) * (np.dot(vel[j] - vel[i], pos[j] - pos[i]) / (np.linalg.norm(pos[j] - pos[i]) ** 2)) * (pos[j] - pos[i])

    # Wall collision
    for i in range(n):
        if pos[i][0] - radii[i] < WALL_SIZE[0]:
            pos[i][0] = 2 * WALL_SIZE[0] + 2 * radii[i] - pos[i][0]
            out_vel[i][0] *= -1
        if pos[i][0] + radii[i] > WALL_SIZE[0] + WALL_GAP[0]:
            pos[i][0] = WALL_SIZE[0] + WALL_GAP[0] - radii[i] + \
                (WALL_SIZE[0] + WALL_GAP[0] - pos[i][0] - radii[i])
            out_vel[i][0] *= -1
        if pos[i][1] - radii[i] < WALL_SIZE[1]:
            pos[i][1] = 2 * WALL_SIZE[1] + 2 * radii[i] - pos[i][1]
            out_vel[i][1] *= -1
        if pos[i][1] + radii[i] > WALL_SIZE[1] + WALL_GAP[1]:
            pos[i][1] = WALL_SIZE[1] + WALL_GAP[1] - radii[i] + \
                (WALL_SIZE[1] + WALL_GAP[1] - pos[i][1] - radii[i])
            out_vel[i][1] *= -1

    out_pos = np.add(pos, ((vel + out_vel) / 2) * dt, out=out_pos)

    return out_pos


def run_simulation(draw=False, save_data=False, num_scenes=1000, max_timesteps=2000):
    """
    Run the simulation for `num_scenes` with `num_timesteps` each scene.
    Properly resets the environment and the physical values each scene.
    """
    # Run simulation
    for curr_scene in trange(num_scenes, desc="Scene"):
        # Reset the environment
        reset_environment()

        # Create save_data folder
        if save_data:
            os.mkdir("{}/{}".format(DATA_FOLDER, curr_scene))

        # Generate values
        v, v2, p, p2, r, m, d, c, collision = generate_initial_values()

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

            delay = 0

        # Run this scene for max_timesteps steps
        for curr_timestep in trange(max_timesteps, desc="Timestep"):
            if draw:
                redraw = False
                while not redraw:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            sys.exit()
                        if event.type == TIME_EVENT_ID:
                            interaction(
                                TIME_DELTA,
                                p,
                                v,
                                m,
                                radii=r,
                                collision=collision,
                                out_pos=p2,
                                out_vel=v2,
                                distance=d,
                                save_data=save_data
                            )

                            # Swap position and velocities
                            p, p2 = p2, p
                            v, v2 = v2, v

                            # End the loop, to be able to redraw the simulation
                            redraw = True
                        # end if
                    # end for

                # Update the timesteps
                delay = (delay + 1) % TIMESTEP_DELAY
                if not delay:
                    for t in range(HIST_TIMESTEPS-1):
                        hp[t, :, :] = hp[t+1, :, :]
                hp[HIST_TIMESTEPS-1, :, :] = p[:, :]

                # Redraw the balls in the simulation
                screen.fill(COLOR_BLACK)
                for t in range(HIST_TIMESTEPS):
                    for i in range(NUM_OF_BODIES):
                        if not (np.isnan(hp[t, i])).any():
                            pygame.gfxdraw.filled_circle(
                                screen,
                                int(hp[t, i, 0] * DRAW_SCALE),
                                int(hp[t, i, 1] * DRAW_SCALE),
                                int(r[i, 0] * DRAW_SCALE),
                                list(c[i]) +
                                [255 // (HIST_TIMESTEPS-t)]
                            )
                    # end for
                # end for

                # Draw rectangles
                for rect in RECTANGLES:
                    pygame.draw.rect(screen, COLOR_GREY, rect)

                # Flip colorbuffer
                pygame.display.flip()
            else:
                if save_data:
                    np.save(DATA_FOLDER + str(curr_scene) + "/" +
                            str(curr_timestep) + '.pos.npy', p)
                    np.save(DATA_FOLDER + str(curr_scene) + "/" +
                            str(curr_timestep) + '.vel.npy', v)
                    np.save(DATA_FOLDER + str(curr_scene) + "/" +
                            str(curr_timestep) + '.mass.npy', m)
                    np.save(DATA_FOLDER + str(curr_scene) + "/" +
                            str(curr_timestep) + '.radii.npy', r)
                    np.save(DATA_FOLDER + str(curr_scene) + "/" +
                            str(curr_timestep) + '.data.npy', np.array([WALL_GAP, WALL_SIZE]))

                interaction(
                    TIME_DELTA,
                    p,
                    v,
                    m,
                    radii=r,
                    collision=collision,
                    out_pos=p2,
                    out_vel=v2,
                    distance=d,
                    save_data=save_data
                )

                # Swap position and velocities
                p, p2 = p2, p
                v, v2 = v2, v
            # end if
        # end for
    # end for


if __name__ == "__main__":
    fire.Fire(run_simulation)
