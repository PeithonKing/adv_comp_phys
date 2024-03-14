import copy
import numpy as np
import matplotlib.pyplot as plt

def wave_eqn(u, delta_t=0.1, delta_x=0.1):
    h = delta_t/delta_x
    u = copy.deepcopy(u)
    Nx = u.shape[0]
    Nt = u.shape[1]

    u[1:Nx-1, 1] = u[1:Nx-1, 0] + (h**2)/2 * (u[0:Nx-2, 0] - 2*u[1:Nx-1, 0] + u[2:Nx, 0])

    for k in range(2, Nt):
        u[1:Nx-1, k] = 2 * u[1:Nx-1, k-1] - u[1:Nx-1, k-2] + (h**2) * (u[0:Nx-2, k-1] - 2*u[1:Nx-1, k-1] + u[2:Nx, k-1])

    return u
    
L, NL = 1, 100
L, NT = 1, 100
dL, dT = L/NL, L/NT

u = np.zeros((NL, NL))

a = 10
b = L/2
c = 0.1

# gaussian pulse
gaussian = lambda x: a * np.exp(-((x - b)**2)/(2*c**2))

u[0, :] = 0
u[1, :] = 0
u[:, 0] = gaussian(np.linspace(0, L, NL))

u = wave_eqn(u, dT, dL)

plt.plot(u[:, 0], label='t=0')
plt.plot(u[:, 10], label='t=10')
plt.plot(u[:, 20], label='t=20')
plt.plot(u[:, 30], label='t=30')
plt.plot(u[:, 40], label='t=40')
plt.plot(u[:, 50], label='t=50')
# plt.plot(u[:, 60], label='t=60')
# plt.plot(u[:, 70], label='t=70')
# plt.plot(u[:, 80], label='t=80')
# plt.plot(u[:, 90], label='t=90')
# plt.plot(u[:, 99], label='t=99')
plt.legend()
plt.show()