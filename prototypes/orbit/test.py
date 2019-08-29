import sys
import time

import numpy as np
import pygame
import pygame.gfxdraw

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (109, 196, 255)

FLOAT_DTYPE = np.float

G = 39.478#6.67408e-11
HIST_TIMESTEPS = 100
TIMESTEP_DELAY = 2
NUM_OF_BODIES = 6
TIME_DELTA = 0.01
TIME_EVENT_ID = pygame.USEREVENT+1
WIDTH = 1620
HEIGHT = 1080

V_INIT_TYPE = "random" # One of: ["random", "circular", "elliptical"]


def nbody(dt, pos, vel, mass, radii=None, out_pos=None, out_vel=None,
          force_placeholder=None, distance_placeholder=None, G=39.478, epsilon=1e-3):
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
                     (epsilon + np.linalg.norm(distance_placeholder[i, j, :]) ** 2))

    out_vel = np.add(vel, force_placeholder / mass * dt, out=out_vel)
    out_pos = np.add(pos, (vel + (out_vel - vel) / 2) * dt, out=out_pos)
    return out_pos, out_vel, force_placeholder


num_dims = 2

v = np.zeros((NUM_OF_BODIES, num_dims), dtype=FLOAT_DTYPE)
v2 = np.zeros((NUM_OF_BODIES, num_dims), dtype=FLOAT_DTYPE)
p2 = np.random.uniform(low=10,
                       high=100, size=(NUM_OF_BODIES, num_dims))
p2[:, 1] = np.random.uniform(low=0, high=2*np.pi, size=NUM_OF_BODIES).astype(FLOAT_DTYPE)
p = np.zeros((NUM_OF_BODIES, num_dims), dtype=FLOAT_DTYPE)
p[:, 0] = p2[:, 0] * np.cos(p2[:, 1])
p[:, 1] = p2[:, 0] * np.sin(p2[:, 1])

m = np.random.uniform(0.02, 9, size=(NUM_OF_BODIES, 1)).astype(FLOAT_DTYPE)
f = np.zeros((NUM_OF_BODIES, num_dims), dtype=FLOAT_DTYPE)
d = np.zeros((NUM_OF_BODIES, NUM_OF_BODIES, num_dims), dtype=FLOAT_DTYPE)

# Configure color
c = np.random.randint(64, 255, size=(NUM_OF_BODIES, 3))

r = np.log2(10*m)

if V_INIT_TYPE == "random":
    v = np.random.uniform(low=-3, high=3, size=(NUM_OF_BODIES,num_dims)).astype(FLOAT_DTYPE)
elif V_INIT_TYPE == "circular":
    # Set up the sun
    p[0, :] = [0, 0]
    m[0, :] = [100]
    r[0, :] = np.log2(m[0,:])
    c[0, :] = [255, 255, 0]
    
    center_of_mass = np.sum(p*m, axis=0)/np.sum(m)
    distance_to_center = np.linalg.norm(p - center_of_mass, axis=1)
    _, _, centripetal_force = nbody(TIME_DELTA, p, v, m, G=G)
    for i in range(NUM_OF_BODIES):
        u_force = centripetal_force[i] / np.linalg.norm(centripetal_force[i])
        velocity_magnitude = np.sqrt(np.linalg.norm(centripetal_force[i])
                             * np.linalg.norm(distance_to_center[i]) / m[i])
        if np.random.choice(["clockwise","counterclockwise"]) == "clockwise":
            v[i, 0] = u_force[1]
            v[i, 1] = -u_force[0]
        else:
            v[i, 0] = -u_force[1]
            v[i, 1] = u_force[0]
        
        v[i] *= velocity_magnitude
        print(velocity_magnitude)
elif V_INIT_TYPE == "elliptical":
    # Set up the sun
    p[0, :] = [0, 0]
    m[0, :] = [100]
    r[0, :] = np.log2(m[0,:])
    c[0, :] = [255, 255, 0]
    
    center_of_mass = np.sum(p*m, axis=0)/np.sum(m)
    distance_to_center = np.linalg.norm(p - center_of_mass, axis=1)
    _, _, centripetal_force = nbody(TIME_DELTA, p, v, m, G=G)
    for i in range(NUM_OF_BODIES):
        u_force = centripetal_force[i] / np.linalg.norm(centripetal_force[i])
        velocity_magnitude = np.sqrt(np.linalg.norm(centripetal_force[i])
                             * np.linalg.norm(distance_to_center[i]) / m[i])
        if np.random.choice(["clockwise","counterclockwise"]) == "clockwise":
            v[i, 0] = u_force[1]
            v[i, 1] = -u_force[0]
        else:
            v[i, 0] = -u_force[1]
            v[i, 1] = u_force[0]
        
        v[i] *= velocity_magnitude * 1.5
        print(velocity_magnitude)
else:
    raise NotImplementedError("No such velocity initilization \"{}\"".format(V_INIT_TYPE))

hp = np.zeros((HIST_TIMESTEPS, NUM_OF_BODIES, num_dims), dtype=FLOAT_DTYPE)
hp[:] = p

pygame.init()
size = WIDTH, HEIGHT
screen = pygame.display.set_mode(size)

pygame.time.set_timer(TIME_EVENT_ID, int(1000*TIME_DELTA))

radius = 0
delay = 0
Umax, Umin = float("-inf"), float("inf")
passed_iter = 0
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
                distance_placeholder=d,
                G=G
            )

            # Swap position and velocities
            p, p2 = p2, p
            v, v2 = v2, v
            
            # Check for energy conservation
            Ug = 0
            for i in range(m.shape[0]):
                for j in range(m.shape[0]):
                    if i != j:
                        Ug -= G*(m[i,0] * m[j,0])/np.linalg.norm(p[i]-p[j])
            Uv = np.sum(np.squeeze(m) * (np.linalg.norm(v, axis=1) ** 2) / 2)
            U = Uv+Ug
            Umax, Umin = max(Umax,U), min(Umin,U)
            passed_iter += 1
            print("Egy U⊥ {Umin:.4e} U⊤ {Umax:.4e} ΔU {Udelta:.4e} U {U:.4e} Uv {Uv:.4e} Ug {Ug:.4e} T {T:.3f}".format(
                    Umin=Umin,
                    Umax=Umax,
                    Udelta=Umax-Umin,
                    U=U,
                    Uv=Uv,
                    Ug=Ug,
                    T=passed_iter*TIME_DELTA
                    ))
        # end if
    # end for

    if redraw:
        delay = (delay + 1) % TIMESTEP_DELAY
        if not delay:
            for t in range(HIST_TIMESTEPS-1):
                hp[t, :, :] = hp[t+1, :, :]
        hp[HIST_TIMESTEPS-1, :, :] = p[:, :]
        
        
        avg_p = np.mean(p,axis=0)
        if V_INIT_TYPE == "circular" or V_INIT_TYPE == "elliptical":
            avg_p = p[0]
        max_p = np.max(np.linalg.norm(p-avg_p[np.newaxis,:],axis=1))
        radius = max(1.5*max_p,radius)
        #radius = 1.5*max_p # Uncomment for trippier visualisation
        if HEIGHT<WIDTH:
            h = 2*radius
            w = h*WIDTH/HEIGHT
        else:
            w = 2*radius
            h = w*HEIGHT/WIDTH

        screen.fill(COLOR_BLACK)
        for t in range(HIST_TIMESTEPS):
            for i in range(NUM_OF_BODIES):
                pygame.gfxdraw.filled_circle(
                    screen,
                    int(WIDTH/2+(hp[t, i, 0]-avg_p[0])*WIDTH/w),
                    int(HEIGHT/2+(hp[t, i, 1]-avg_p[1])*HEIGHT/h),
                    int(r[i, 0]*min(WIDTH,HEIGHT)/radius),
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
