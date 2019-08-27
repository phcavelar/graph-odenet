import sys
import time

import numpy as np
import pygame
import pygame.gfxdraw

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (109, 196, 255)

# Multiply by a big constant, otherwise the bodies would not move given
# the small value of the gravitational constant
G = 39.478
HIST_TIMESTEPS = 100
TIMESTEP_DELAY = 2
NUM_OF_BODIES = 5
TIME_DELTA = 0.01
TIME_EVENT_ID = pygame.USEREVENT+1
WIDTH = 1620
HEIGHT = 1080


def nbody(dt, pos, vel, mass, radii=None, out_pos=None, out_vel=None,
          force_placeholder=None, distance_placeholder=None, G=39.478):
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
                    (G * m[i] * m[j] /
                     (np.linalg.norm(distance_placeholder[i, j, :]) ** 2))

    out_vel = np.add(vel, force_placeholder / mass * dt, out=out_vel)
    out_pos = np.add(pos, (vel + (out_vel - vel) / 2) * dt, out=out_pos)
    return out_pos, out_vel, force_placeholder


num_dims = 2

v = np.zeros((NUM_OF_BODIES, num_dims), dtype=np.float)
v2 = np.zeros((NUM_OF_BODIES, num_dims), dtype=np.float)
p2 = np.random.uniform(low=min(WIDTH, HEIGHT)*1/7,
                       high=min(WIDTH, HEIGHT)*1/4, size=(NUM_OF_BODIES, num_dims))
p2[:, 1] = np.random.uniform(low=0, high=2*np.pi, size=NUM_OF_BODIES)
p = np.zeros((NUM_OF_BODIES, num_dims), dtype=np.float)
p[:] = np.array([WIDTH/2, HEIGHT/2])
p[:, 0] += p2[:, 0] * np.cos(p2[:, 1])
p[:, 1] += p2[:, 0] * np.sin(p2[:, 1])

m = np.random.uniform(2, 5, size=(NUM_OF_BODIES, 1))
f = np.zeros((NUM_OF_BODIES, num_dims), dtype=np.float)
d = np.zeros((NUM_OF_BODIES, NUM_OF_BODIES, num_dims), dtype=np.float)

# Configure color
c = np.random.randint(0, 255, size=(NUM_OF_BODIES, 3))

# Set up the sun
p[0, :] = [WIDTH/2, HEIGHT/2]
m[0, :] = [np.sum(m[1:])/0.01]
c[0, :] = [255, 255, 0]

print("pol", p2.shape, p2)
print("car", p.shape, p)

r = np.log2(m**2)
r[1:] *= 3

hp = np.zeros((HIST_TIMESTEPS, NUM_OF_BODIES, num_dims), dtype=np.float)
hp[:] = p

center_of_mass = np.sum(p*m, axis=0)/np.sum(m)
distance_to_center = np.linalg.norm(p - center_of_mass, axis=1)
_, _, centripetal_force = nbody(TIME_DELTA, p, v, m)
for i in range(NUM_OF_BODIES):
    v[i, 0] = centripetal_force[i, 1] / np.linalg.norm(centripetal_force[i])
    v[i, 0] *= np.sqrt(np.linalg.norm(centripetal_force[i])
                       * np.linalg.norm(distance_to_center[i]) / m[i])
    v[i, 1] = centripetal_force[i, 0] / np.linalg.norm(centripetal_force[i])
    v[i, 1] *= np.sqrt(np.linalg.norm(centripetal_force[i])
                       * np.linalg.norm(distance_to_center[i]) / m[i])


pygame.init()
size = WIDTH, HEIGHT
screen = pygame.display.set_mode(size)

font = pygame.font.SysFont('Arial', 16)
text = font.render('0', True, COLOR_BLUE)
textRect = text.get_rect()
pygame.time.set_timer(TIME_EVENT_ID, int(1000*TIME_DELTA))

delay = 0
bla, ble = float("-inf"), float("inf")
while True:
    redraw = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == TIME_EVENT_ID:
            redraw = True
            nbody(
                TIME_DELTA,
                p,
                v,
                m,
                out_pos=p2,
                out_vel=v2,
                force_placeholder=f,
                distance_placeholder=d
            )

            # Swap position and velocities
            p, p2 = p2, p
            v, v2 = v2, v

            #pairshape = [m.shape[0], m.shape[0]]
            # pairmask = np.ones(
            #    pairshape+[1], dtype=np.int) - np.eye(m.shape[0], dtype=np.int)[..., np.newaxis]
            Ug = 0
            for i in range(m.shape[0]):
                for j in range(m.shape[0]):
                    if i != j:
                        Ug -= G*(m[i] * m[j])/np.linalg.norm(p[i]-p[j])

            Uv = np.sum(np.squeeze(m) * (np.linalg.norm(v, axis=1) ** 2) / 2)
            # Ug = np.sum(-G*(m.view()[np.newaxis, :, :] * m.view()[:, np.newaxis, :])[
            #            pairmask]/(p.view()[np.newaxis, :, :]-p.view()[:, np.newaxis, :])[pairmask])

            # bla = max(
            #    np.sum(np.squeeze(m) * (np.linalg.norm(v, axis=1) ** 2) / 2), bla)
            # ble = min(
            #    np.sum(np.squeeze(m) * (np.linalg.norm(v, axis=1) ** 2) / 2), ble)
            print(Uv+Ug, Uv, Ug)
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
