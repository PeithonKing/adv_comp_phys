from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

class BasicVarlet:
    def __init__(
        self,
        x0: int | float,
        v0: int | float,
        a: Callable,
        dt=0.01,
        t0=0
    ):
        self.xs = np.array([x0])
        self.vs = np.array([v0])
        self.a = a
        self.dt = dt
        self.t = t0
        print(f"{self.xs.shape = }, {self.vs.shape = }, {self.a = }, {self.dt = }, {self.t = }")

    def _proceed_a_step(self):
        try:
            new_x = 2*self.xs[-1] - self.xs[-2] + self.a(self.xs[-1]) * self.dt**2
        except IndexError:
            new_x = self.xs[-1] + self.vs[-1] * self.dt + 0.5 * self.a(self.xs[-1]) * self.dt**2
        new_v = (new_x - self.xs[-1]) / self.dt

        self.xs = np.append(self.xs, [new_x], axis=0)
        self.vs = np.append(self.vs, [new_v], axis=0)
        self.t += self.dt

    def integrate(self, total_time):
        # Calculate the number of steps based on total time and time step
        num_steps = int(total_time / self.dt)

        # Perform the specified number of time steps
        for _ in range(num_steps):
            self._proceed_a_step()

        return self.xs, self.vs

class VelocityVarlet:
    def __init__(
        self,
        x0: int | float,
        v0: int | float,
        a: Callable,
        dt=0.01,
        t0=0
    ):
        self.xs = np.array([x0])
        self.vs = np.array([v0])
        self.a = a
        self.dt = dt
        self.t = t0

    def _proceed_a_step(self):
        v_half = self.vs[-1] + 0.5 * self.a(self.t) * self.dt**2
        new_x = self.xs[-1] + v_half * self.dt
        new_v = self.vs[-1] + 0.5 * (self.a(self.t) + self.a(self.t + self.dt)) * self.dt
        
        self.xs = np.append(self.xs, [new_x], axis=0)
        self.vs = np.append(self.vs, [new_v], axis=0)

    def integrate(self, total_time):
        # Calculate the number of steps based on total time and time step
        num_steps = int(total_time / self.dt)

        # Perform the specified number of time steps
        for _ in range(num_steps):
            self._proceed_a_step()
        
        return self.xs, self.vs

class Leapfrog:
    def __init__(
        self,
        x0: int | float,
        v0: int | float,
        a: Callable,
        dt=0.01,
        t0=0
    ):
        self.xs = np.array([x0])
        self.vs = np.array([v0])
        self.a = a
        self.dt = dt
        self.t = t0
        print(f"{self.xs.shape = }, {self.vs.shape = }, {self.a = }, {self.dt = }, {self.t = }")

    def _proceed_a_step(self):
        v_half = self.vs[-1] + 0.5 * self.a(self.xs[-1]) * self.dt
        new_x = self.xs[-1] + v_half * self.dt
        new_v = v_half + 0.5 * self.a(new_x) * self.dt

        self.xs = np.append(self.xs, [new_x], axis=0)
        self.vs = np.append(self.vs, [new_v], axis=0)
        self.t += self.dt

    def integrate(self, total_time):
        # Calculate the number of steps based on total time and time step
        num_steps = int(total_time / self.dt)

        # Perform the specified number of time steps
        for _ in range(num_steps):
            self._proceed_a_step()

        return self.xs, self.vs
            

# Example usage
initial_positions = [0, 0, 0]
initial_velocities = [5, 0, 5]

a = lambda x: np.array([0, 0, -10])
dt = 0.01

# Create an instance of VarletIntegrator
# integrator = BasicVarlet(initial_positions, initial_velocities, a, dt)
# integrator = VelocityVarlet(initial_positions, initial_velocities, a, dt)
integrator = Leapfrog(initial_positions, initial_velocities, a, dt)

# Integrate over a total time of 1.0 seconds
integrator.integrate(1.0)

# Plot the results
plt.plot(integrator.xs[:, 2], label="x")
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Position")
plt.show()