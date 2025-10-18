import gymnasium as gym
import numpy as np
from gymnasium import logger
from gymnasium.error import DependencyNotInstalled


class DoublePendulumEnv(gym.Env):
    def __init__(
        self,
        dynamics_func,
        reset_func,
        obs_space=gym.spaces.Box(
            np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
        ),
        mode="human",
        act_space=gym.spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        max_episode_steps=1000,
        disturbance = False,
        unscaled_constraints = None,
        theta_slack = 0,
        penalty_k = 0,
        terminates=True,
        render_fps=60,
    ):
        self.dynamics_func = dynamics_func
        self.reset_func = reset_func
        self.observation_space = obs_space
        self.action_space = act_space
        self.max_episode_steps = max_episode_steps

        self.max_V = self.dynamics_func.simulator.plant.potential_energy(
            self.dynamics_func.unscale_state(np.array([0.0, -1.0, 0.0, 0.0]))
        )

        self.terminates = terminates
        # For rendering
        self.mode = mode
        self.render_fps = render_fps
        self.SCREEN_DIM = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.observation = self.reset_func()
        self.step_counter = 0
        self.stabilisation_mode = False
        self.y = [0.0, 0.0]

        self.disturbance = disturbance
        self.disturbance_activation_step = 100
        self.disturbance_length = 100
        self.unscaled_constraints = unscaled_constraints
        if self.unscaled_constraints is not None:
            self.constr_th1, self.constr_th2, _, _ = self.unscaled_constraints
        self.theta_slack = theta_slack
        self.penalty_k = penalty_k

        self.theta1_buffer = []
        self.theta2_buffer = []

        l1 = self.dynamics_func.simulator.plant.l[0]
        l2 = self.dynamics_func.simulator.plant.l[1]
        self.max_height = l1 + l2

        if self.dynamics_func.robot == "acrobot":
            self.control_line = 0.92 * self.max_height
        elif self.dynamics_func.robot == "pendubot":
            self.control_line = 0.7 * self.max_height

    # Update the y coordinate of the first joint and the end effector
    def update_y(self):
        theta1, theta2, _, _ = self.dynamics_func.unscale_state(self.observation)

        link_end_points = self.dynamics_func.simulator.plant.forward_kinematics(
            [theta1, theta2]
        )
        self.y[0] = link_end_points[0][1]
        self.y[1] = link_end_points[1][1]

    def gravitational_reward(self):
        x = self.dynamics_func.unscale_state(self.observation)
        V = self.dynamics_func.simulator.plant.potential_energy(x)
        return V

    def V(self):
        return self.gravitational_reward()

    def kinetic_reward(self):
        x = self.dynamics_func.unscale_state(self.observation)
        T = self.dynamics_func.simulator.plant.kinetic_energy(x)
        return T

    def T(self):
        return self.kinetic_reward()

    def reward_func(self, terminated):
        theta1, theta2, _, _ = self.dynamics_func.unscale_state(self.observation)
        costheta2 = np.cos(theta2)

        if not terminated:
            if self.dynamics_func.robot == "acrobot":
                if self.stabilisation_mode:
                    # reward = self.max_V + self.V() + (1 + costheta2) ** 2 - 2 * self.T()
                    reward = self.V() + 2 * (1 + costheta2) ** 2 - 2 * self.T()
                else:
                    # reward = np.max([self.V() - 1.5 * self.y[1] * self.T(), 0])
                    reward = self.V()
#
            elif self.dynamics_func.robot == "pendubot":
                if self.stabilisation_mode:
                    reward = self.V() + 2 * (1 + costheta2) ** 2 - 2 * self.T()
                else:
                    reward = self.V()
            if self.unscaled_constraints is not None:
                curr_theta1 = abs(np.unwrap(self.theta1_buffer)[-1])
                curr_theta2 = abs(np.unwrap(self.theta2_buffer)[-1])
                epsilon = 0.001
                
                if curr_theta1 > self.constr_th1 - epsilon:
                    curr_theta1 = self.constr_th1 - epsilon
                if curr_theta2 > self.constr_th2 - epsilon:
                    curr_theta2 = self.constr_th2 - epsilon
                
                penalty_th1 = 0.5*4*np.log(-curr_theta1+self.constr_th1)
                penalty_th2 = 0.5*4*np.log(-curr_theta2+(self.constr_th2 + self.theta_slack))
                if self.stabilisation_mode:
                    penalty_th2 = self.penalty_k*penalty_th2 + 2*np.cos(curr_theta2) # pi2 has +pi/2. 0.5 here is only for pi/2
                
                # print(reward, penalty_th1, penalty_th2)
                reward = reward + penalty_th1 + penalty_th2
        else:
            reward = -1.0
        return reward

    def terminated_func(self):
        if self.terminates:
            # Checks if we're in stabilisation mode and the ee has fallen below the control line
            if self.stabilisation_mode and self.y[1] < self.control_line:
                return True

        return False
    # problem is in unscale in their library: here the unscale is correct given that -1 -1 0 0 is downwards. On the other one it is not correct.
    def step(self, action):

        if self.disturbance and self.step_counter == self.disturbance_activation_step:
            action = np.array([self.dynamics_func.torque_limit[0]*0.75], dtype=np.float32)

        if self.disturbance and self.step_counter > self.disturbance_activation_step and self.step_counter <= self.disturbance_activation_step + self.disturbance_length:
            action = np.array([0], dtype=np.float32)

        self.observation = self.dynamics_func(self.observation, action)
        th1, th2, _, _ = self.dynamics_func.unscale_state(self.observation)
        self.theta1_buffer.append(th1)
        self.theta2_buffer.append(th2)

        # print(action)
        self.update_y()
        # print(self.y[1], self.control_line)

        if self.y[1] >= self.control_line:
            self.stabilisation_mode = True

        terminated = self.terminated_func()

        if self.y[1] < self.control_line:
            self.stabilisation_mode = False # for rewards during recovery

        reward = self.reward_func(terminated)

        truncated = False
        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            self.step_counter = 0

        self.previous_action = action[0]
        return self.observation, reward, terminated, truncated, {}

    def reset(self, seed=0, options=None):
        super().reset(seed=seed)

        self.previous_action = 0
        self.observation = self.reset_func()
        self.step_counter = 0
        self.stabilisation_mode = False
        self.theta1_buffer = []
        self.theta2_buffer = []
        return self.observation, {}

    def render(self):
        if self.mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the mode at initialization, "
                f'e.g. gym("{self.spec.id}", mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))

        l1 = self.dynamics_func.simulator.plant.l[0]
        l2 = self.dynamics_func.simulator.plant.l[1]

        bound = l1 + l2 + 0.2  # 2.2 for default
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        # s = self.scale_state()
        s = self.dynamics_func.unscale_state(self.observation)

        if s is None:
            return None

        p1 = [
            -l1 * np.cos(s[0]) * scale,
            l1 * np.sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - l2 * np.cos(s[0] + s[1]) * scale,
            p1[1] + l2 * np.sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2]
        link_lengths = [l1 * scale, l2 * scale]

        for (x, y), th, llen in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            t, b = 0.02 * scale, -0.02 * scale

            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))

            gfxdraw.filled_circle(
                surf, int(x), int(y), int(0.03 * scale), (204, 204, 0)
            )

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        # Display angle information
        pygame.font.init()
        font = pygame.font.SysFont("Comic Sans MS", 30)
        text_surface = font.render(
            f"Angle 1: {int(s[0] * 180/ np.pi) } | Angle 2: {int(s[1] * 180/np.pi)}",
            False,
            (0, 0, 0),
        )
        self.screen.blit(text_surface, (0, 0))

        if self.mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

        elif self.mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class double_pendulum_dynamics_func:
    def __init__(
        self,
        simulator,
        dt=0.01,
        integrator="runge_kutta",
        robot="acrobot",
        state_representation=2,
        max_velocity=20.0,
        torque_limit=[10.0, 10.0],
    ):
        self.simulator = simulator
        self.dt = dt
        self.integrator = integrator
        self.robot = robot
        self.state_representation = state_representation
        self.max_velocity = max_velocity
        self.torque_limit = torque_limit

    def __call__(self, state, action):
        x = self.unscale_state(state)
        u = self.unscale_action(action)
        xn = self.integration(x, u)
        obs = self.normalize_state(xn)
        return np.array(obs, dtype=np.float32)

    def integration(self, x, u):
        if self.integrator == "runge_kutta":
            next_state = np.add(
                x,
                self.dt * self.simulator.runge_integrator(x, self.dt, 0.0, u),
                casting="unsafe",
            )
        elif self.integrator == "euler":
            next_state = np.add(
                x,
                self.dt * self.simulator.euler_integrator(x, self.dt, 0.0, u),
                casting="unsafe",
            )
        return next_state

    def unscale_action(self, action):
        """
        scale the action
        [-1, 1] -> [-limit, +limit]
        """
        if self.robot == "double_pendulum":
            a = [
                float(self.torque_limit[0] * action[0]),
                float(self.torque_limit[1] * action[1]),
            ]
        elif self.robot == "pendubot":
            a = np.array([float(self.torque_limit[0] * action[0]), 0.0])
        elif self.robot == "acrobot":
            a = np.array([0.0, float(self.torque_limit[1] * action[0])])
        return a

    def unscale_state(self, observation):
        """
        scale the state
        [-1, 1] -> [-limit, +limit]
        """
        if self.state_representation == 2:
            x = np.array(
                [
                    observation[0] * np.pi + np.pi,
                    observation[1] * np.pi + np.pi,
                    observation[2] * self.max_velocity,
                    observation[3] * self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            x = np.array(
                [
                    np.arctan2(observation[0], observation[1]),
                    np.arctan2(observation[2], observation[3]),
                    observation[4] * self.max_velocity,
                    observation[5] * self.max_velocity,
                ]
            )
        return x

    def normalize_state(self, state):
        """
        rescale state:
        [-limit, limit] -> [-1, 1]
        """
        if self.state_representation == 2:
            observation = np.array(
                [
                    (state[0] % (2 * np.pi) - np.pi) / np.pi,
                    (state[1] % (2 * np.pi) - np.pi) / np.pi,
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            observation = np.array(
                [
                    np.cos(state[0]),
                    np.sin(state[0]),
                    np.cos(state[1]),
                    np.sin(state[1]),
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )

        return observation
