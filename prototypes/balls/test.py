import sys
import time

import numpy as np
import pygame
import pygame.gfxdraw

# Dev
from pprint import pprint as pp

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (109, 196, 255)

HIST_TIMESTEPS = 100
TIMESTEP_DELAY = 1
NUM_OF_BODIES = 15
TIME_DELTA = 0.01
TIME_EVENT_ID = pygame.USEREVENT+1
WIDTH = 1920
HEIGHT = 1080
WALL_FRACTION = 1/12
FONT_SIZE = 32


def interaction(dt, pos, vel, mass, radii=None, collision=None, out_pos=None, out_vel=None,
                force=None, distance=None):
    collision, out_pos, out_vel, force = map(lambda x: np.empty_like(
        pos) if x is None else x, (collision, out_pos, out_vel, force))

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
        if pos[i][0] - radii[i] < WIDTH * WALL_FRACTION:
            pos[i][0] = WIDTH * WALL_FRACTION + radii[i] - \
                (pos[i][0] - radii[i] - WIDTH * WALL_FRACTION)
            out_vel[i][0] *= -1
        if pos[i][0] + radii[i] > WIDTH * (1 - WALL_FRACTION):
            pos[i][0] = WIDTH * (1 - WALL_FRACTION) - radii[i] + \
                (WIDTH * (1 - WALL_FRACTION) - pos[i][0] - radii[i])
            out_vel[i][0] *= -1
        if pos[i][1] - radii[i] < HEIGHT * WALL_FRACTION:
            pos[i][1] = HEIGHT * WALL_FRACTION + radii[i] - \
                (pos[i][1] - radii[i] - HEIGHT * WALL_FRACTION)
            out_vel[i][1] *= -1
        if pos[i][1] + radii[i] > HEIGHT * (1 - WALL_FRACTION):
            pos[i][1] = HEIGHT * (1 - WALL_FRACTION) - radii[i] + \
                (HEIGHT * (1 - WALL_FRACTION) - pos[i][1] - radii[i])
            out_vel[i][1] *= -1

    out_pos = np.add(pos, ((vel + out_vel) / 2) * dt, out=out_pos)

    return out_pos, out_vel, force


num_dims = 2

v = np.random.uniform(-250, 250, size=(NUM_OF_BODIES, num_dims))
v2 = np.copy(v)
p = np.random.uniform(low=min(WIDTH, HEIGHT) * WALL_FRACTION,
                      high=min(WIDTH, HEIGHT) * (1 - WALL_FRACTION), size=(NUM_OF_BODIES, num_dims))
p2 = np.copy(p)

r = m = np.random.uniform(5, 50, size=(NUM_OF_BODIES, 1))
f = np.zeros((NUM_OF_BODIES, num_dims), dtype=np.float)
d = np.zeros((NUM_OF_BODIES, NUM_OF_BODIES), dtype=np.float)
collision = np.zeros((NUM_OF_BODIES, NUM_OF_BODIES), dtype=np.float)

# Configure color
c = np.random.randint(0, 255, size=(NUM_OF_BODIES, 3))

# Configure history
hp = np.empty((HIST_TIMESTEPS, NUM_OF_BODIES, num_dims), dtype=np.float)
hp[:] = np.nan

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

pygame.time.set_timer(TIME_EVENT_ID, int(1000*TIME_DELTA))

delay = 0
while True:
    redraw = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == TIME_EVENT_ID:
            redraw = True
            interaction(
                TIME_DELTA,
                p,
                v,
                m,
                radii=r,
                collision=collision,
                out_pos=p2,
                out_vel=v2,
                force=f,
                distance=d
            )

            # Swap position and velocities
            p, p2 = p2, p
            v, v2 = v2, v
        # end if
    # end for

    if redraw:
        delay = (delay + 1) % TIMESTEP_DELAY
        if not delay:
            for t in range(HIST_TIMESTEPS-1):
                hp[t, :, :] = hp[t+1, :, :]
        hp[HIST_TIMESTEPS-1, :, :] = p[:, :]

        screen.fill(COLOR_BLACK)
        for t in range(HIST_TIMESTEPS):
            for i in range(NUM_OF_BODIES):
                if not (np.isnan(hp[t, i])).any():
                    pygame.gfxdraw.filled_circle(
                        screen,
                        int(hp[t, i, 0]),
                        int(hp[t, i, 1]),
                        int(r[i, 0]),
                        list(c[i]) + [255 // (HIST_TIMESTEPS-t)]
                    )

                    # Draw Solid circles without alpha
                    # pygame.draw.circle(
                    #        screen,
                    #        list(c[i]) + [255/(HIST_TIMESTEPS-t)],
                    #        list(map(int,(hp[t,i,0], hp[t,i,1]))),
                    #        int(r[i])
                    # )
        # end for
        pygame.display.flip()
