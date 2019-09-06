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
SUN_MASS = 100
MIN_POSITION, MAX_POSITION = (10, 100)

# Visualization constants
HIST_TIMESTEPS = 200
TIMESTEP_DELAY = 10
DATA_FOLDER = './data/{}'.format(NUM_OF_BODIES)


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


def compute_radius(m):
    r = np.log2(2 * m)

    # Check sun(s) radius
    if abs(m[SUN_INDEX, :] - SUN_MASS) < 0.001:
        r[SUN_INDEX, :] = 2 * np.log2(np.sum(r[SUN_INDEX + 1:, :]))

    return r


def replay_simulation(folds: int = 10):
    """
    Replay a simulation saved.
    """
    REPLAY_FOLDER = './output'

    for replay_file in [x for x in os.listdir(REPLAY_FOLDER) if ".git" not in x]:
        replay_np = np.load("{}/{}".format(REPLAY_FOLDER, replay_file))

        pygame.init()
        # screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        screen = pygame.display.set_mode((WIDTH, HEIGHT))

        pygame.time.set_timer(TIME_EVENT_ID, int(10000*TIME_DELTA))

        # Configure variables to draw
        delay = 0
        radius = 0

        # Configure history
        hp = np.empty((HIST_TIMESTEPS, replay_np.shape[1],
                       NUM_DIMS), dtype=np.float)
        hp[:] = np.nan

        c = np.random.randint(85, 255, size=(replay_np.shape[1], 3))

        old_p = None
        
        first_position = replay_np[0]
        vx_p, vy_p, px_p, py_p, m_p = np.split(replay_np[0],replay_np[0].shape[-1],-1)
        r_p = compute_radius(m_p)
        v_p = np.concatenate([vx_p,vy_p], axis=-1)
        p_p = np.concatenate([px_p,py_p], axis=-1)
        
        for bodies in replay_np:
            vx, vy, px, py, m = np.split(bodies,bodies.shape[-1],-1)
            r = compute_radius(m)
            v = np.concatenate([vx,vy], axis=-1)
            p = np.concatenate([px,py], axis=-1)
            
            pass_next = False
            redraw = False
            while not redraw:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        pass_next = True
                        redraw = True
                    if event.type == TIME_EVENT_ID:

                        p_p, v_p, _ = nbody(
                            TIME_DELTA,
                            p_p,
                            v_p,
                            m_p,
                            G=G
                        )

                        # End the loop, to be able to redraw the simulation
                        redraw = True
                    # end if
                # end for
            # end while
            
            if pass_next:
              break

            # Update the timesteps
            delay = (delay + 1) % TIMESTEP_DELAY
            if not delay:
                for t in range(HIST_TIMESTEPS-1):
                    hp[t, :, :] = hp[t+1, :, :]
            hp[HIST_TIMESTEPS-1, :, :] = p[:, :]

            # Recompute the max_p and radius, used to configure zoom
            avg_p = [0,0]#np.mean(p, axis=0)
            #max_p = np.max(np.linalg.norm(
            #    p-avg_p[np.newaxis, :], axis=1))
            #radius = max(1.5 * max_p, radius)
            #radius = 1.5*max_p # Uncomment for trippier visualisation

            #if HEIGHT < WIDTH:
            #    h = 2 * radius
            #    w = h * WIDTH/HEIGHT
            #else:
            #    w = 2 * radius
            #    h = w * HEIGHT/WIDTH
            radius = 80
            h = 2 * radius
            w = h * WIDTH/HEIGHT

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
                            int(max(1, r[i, 0] *
                                    min(WIDTH, HEIGHT) / radius)),
                            list(c[i]) + [255 // (HIST_TIMESTEPS-t)]
                        )
            
            for i in range(NUM_OF_BODIES):
                  pygame.gfxdraw.filled_circle(
                      screen,
                      int(WIDTH/2 +
                          (p_p[i, 0] - avg_p[0]) * WIDTH/w),
                      int(HEIGHT/2 + (p_p[i, 1] -
                                      avg_p[1]) * HEIGHT/h),
                      int(max(1, r[i, 0] *
                              min(WIDTH, HEIGHT) / radius)),
                      [255, 255, 255, 100]
                  )

            # Flip color buffer
            pygame.display.flip()

            # Save old p
            old_p = p.copy()

        # end for
    # end for


if __name__ == "__main__":
    fire.Fire(replay_simulation)