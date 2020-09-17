#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Easy generation of a toy dataset: Lorenz attractor

The toy dataset comprise several trajectories
of a Lorenz chaotic attractor. Each trajectory
can be understood as a class/target category
with several sampled points along it.

The Lorenz system obeys the following ordinary
differential equation:
x' = sigma * (y - x)
y' = rho * x - y - x * z
z' = x * y - beta * z
with a given initial state x0, y0, z0.

A trajectory sample consists of:
1. An initial state
2. A sequence strictly increasing timestamps at which
   to solve the system.

The timestamps can be generated via an input distribution of
time *intervals*. Resulting timestamps are the cumulative sum
of those intervals.

Final dataset has the following layout:
X - data -> Array of shape (n_samples, 4) (t,x,y,z)
Y - target -> Array of shape (n_samples, 7):
  traj. id, sigma, rho, beta, x0, y0, z0

This file also implements a self-contained command line tool
to generate quickly a dataset.
"""

from typing import Callable, Sequence, Tuple
import numpy as np
from scipy.integrate import odeint

# Sampler type alias, deterministic or randomized
Sampler = Callable[[], float]


class LorenzDatasetGenerator:
    """ Lorenz system-based dataset generator


    Dataset is geenrated by solving the ODE system
    for various random initial conditions at random
    varying timesteps.
    """

    def __init__(
            self,
            sigma: float,
            rho: float,
            beta: float,
            seed=None) -> None:
        """ Create new dataset generator

        Seed paramter can be used to set Numpy's PRNG in a
        known state, hereby reproducing the exact same samples
        as a previous invocation with same seed.
        """
        # Sanity checks
        if sigma <= 0:
            raise ValueError(
                f'Parameter {sigma} should be a strictly positive float')
        if rho <= 0:
            raise ValueError(
                f'Parameter {rho} should be a strictly positive float')
        if beta <= 0:
            raise ValueError(
                f'Parameter {beta} should be a strictly positive float')

        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        # Seed numpy PRNG, used for repoduceability
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def generate_timestamps(timer: Sampler,
                            n_samples: int,
                            lowest_resolution: float) -> np.array:
        """ Generate timestamp sequence

        From a distribution of time intervals, create
        the seuqnece of timestamp at which to solve the
        Lorenz system.

        Params:
        timer: time interval sampler
        n_samples: number of samples to draw

        Return:
        strictly increasing sequence of timestamps
        """
        return np.cumsum([max(lowest_resolution, timer())
                          for _ in range(n_samples)],
                         dtype=float)

    def generate_trajectory(self,
                            init_state: np.array,
                            timestamps: Sequence) -> Sequence:
        """ Sample a given trajectory summary

        Params:
        init_state: trajectory specification
        timestamps: sequence of time to solve the system

        Return:
        Sequence of system state at desired timestamps
        """
        assert init_state.size == 3

        # Lorenz system
        def lorenz_system(state, curtime):  # pylint: disable=unused-argument
            """ Lorenz ODE system """
            x, y, z = state[0], state[1], state[2]
            dx = self.sigma * (y - x)
            dy = self.rho * x - y - x * z
            dz = x * y - self.beta * z
            return np.array([dx, dy, dz])

        return odeint(lorenz_system, init_state, timestamps)

    def generate_dataset(self,
                         init_states: np.array,
                         n_samples_per_trajectory: int,
                         timestamp: Sampler,
                         lowest_resolution: float = 1e-4
                         ) -> Tuple[np.array, np.array]:
        """ Generate a Lorenz dataset

        Params:
        init_states: array of shape (n_trajectories, 3)
        n_samples_per_trajectory: self-explanatory
        timestamp: distribution of time interval at which to solve the sytem

        Return: (X, Y) where
        X: array of shape (n_traj * n_samples_per_traj, 4) t,x,y,z
        Y: array of shape (n_traj * n_samples_per_traj, 7)
           (id, sigma,rho,beta,x0,y0,z0)
        """
        n_trajectories = len(init_states)
        n_samples = n_trajectories * n_samples_per_trajectory
        X = np.zeros((n_samples, 4))  # t, x,y,z dimension
        Y = np.zeros((n_samples, 7))  # id, Lorenz system param & initial state

        for i, t in enumerate(range(0, n_samples, n_samples_per_trajectory)):
            initial_state = init_states[i]

            # Generate timestamps
            timestamps = self.generate_timestamps(timestamp,
                                                  n_samples_per_trajectory,
                                                  lowest_resolution)

            # Compute trajectory samples
            samples = self.generate_trajectory(initial_state, timestamps)

            # Update X
            X[t:t+n_samples_per_trajectory, 0] = timestamps
            X[t:t+n_samples_per_trajectory, 1:] = samples

            # Update Y with trajectory ID and parameters
            target = np.array([i, self.sigma, self.rho, self.beta])
            target = np.hstack([target, initial_state]).reshape((1, 7))
            Y[t:t+n_samples_per_trajectory, :] = \
                np.ones((n_samples_per_trajectory, 1)) @ target

        return X, Y


if __name__ == "__main__":
    import pandas as pd
    import argparse

    def check_lorenz(value: str) -> float:
        """ Check basic Lorenz system input """
        fvalue = float(value)
        if fvalue <= 0:
            raise ValueError("Strictly positive float expected")
        return fvalue

    def check_integer(value: str) -> int:
        """ Check parameters' naturality """
        ivalue = int(value)
        ivalue = max(1, ivalue)
        return ivalue

    parser = argparse.ArgumentParser(
        description='Generate Lorenz system dataset')
    parser.add_argument('filename', type=str,
                        help='Filename to store CSV output')

    # pylint: disable=protected-access
    optional = parser._action_groups.pop()

    required = parser.add_argument_group('required arguments')
    required.add_argument('--n-trajectories',
                          type=check_integer, required=True,
                          help='Number of trajectories to sample')
    required.add_argument('--n-samples-per-trajectory',
                          type=check_integer, required=True,
                          help='Number of samples per trajectory')
    required.add_argument(
        '--sampling-frequency', type=check_lorenz, default=10,
        help='Number of samples per second to register')

    parser._action_groups.append(optional)
    optional.add_argument('--sigma', type=check_lorenz, default=10.,
                          help='sigma: strictly positive real')
    optional.add_argument('--rho', type=check_lorenz, default=28.,
                          help='rho: strictly positive real')
    optional.add_argument('--beta', type=check_lorenz, default=8./3.,
                          help='beta: strictly positive real')
    optional.add_argument('--seed', type=int, default=None,
                          help='PRNG seed')

    args = parser.parse_args()
    gen = LorenzDatasetGenerator(args.sigma, args.rho, args.beta, args.seed)

    initial_states = np.vstack([10 * np.random.random((3,)) - 5.
                                for _ in range(args.n_trajectories)])

    raw_data = gen.generate_dataset(initial_states,
                                    args.n_samples_per_trajectory,
                                    lambda: 1./args.sampling_frequency)
    COLS = ["T", "X", "Y", "Z",
            "TID", "rho", "sigma", "beta",
            "X_0", "Y_0", "Z_0"]
    dataset = pd.DataFrame(np.hstack(raw_data),
                           columns=COLS)
    dataset.index.name = "SID"  # Sample ID
    dataset.to_csv(args.filename)
