import argparse
import numpy as np
from scipy.constants import Boltzmann

parser = argparse.ArgumentParser()
parser.add_argument("-E", default = np.array([0.0, 0.0, 0.0]), nargs = '+', type = np.array, help="Electric field")
parser.add_argument("-B", default = np.array([0.0, 0.0, 0.0]), nargs = '+', type = np.array, help="Magnetic field")
parser.add_argument("-T", default = 0, type = float, help = "Initial temperature")
parser.add_argument("-q", default = 1, type = float, help = "Charge")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--layers", default=1, type=int, help="Number of layers.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--forces", default='', type=str, help='Source of forces file')
parser.add_argument("--energies", default='', type=str, help='Source of energies file')
parser.add_argument("--movie", default='', type=str, help='Source of movie file')
parser.add_argument("--load", default=False, type=bool, help='Load new data')

mass_of_argon = np.array([39.948])  # amu

def init_velocity(T, number_of_particles):
    R = np.random.rand(number_of_particles, 3) - 0.5
    return R * np.sqrt(Boltzmann * T / (mass_of_argon[0] * 1.602e-19))/3

# def lj_force(r, epsilon, sigma):
#     return 48 * epsilon * np.power(sigma, 12) / np.power(r, 13) - 24 * epsilon * np.power(sigma, 6) / np.power(r, 7)

def get_accelerations(positions, velocity, args):
    # párová interakce částic
    accel_x = np.zeros((3, len(positions), len(positions)))
    # for i in range(0, len(positions) - 1):
    #     for j in range(i + 1, len(positions)):
    #         r_x = positions[j] - positions[i]
    #         rmag = np.linalg.norm(r_x)
    #         force_scalar = lj_force(rmag, 0.0103, 3.4)
    #         force_x = force_scalar * r_x / rmag
    #         for x in range(3):
    #             accel_x[x, i, j] = - force_x[x] / mass_of_argon
    #             accel_x[x, j, i] = force_x[x] / mass_of_argon
    a = np.sum(accel_x.T, axis=0)

    # interakce externího pole
    for i in range(0, len(positions)):
        F = args.q*(args.E + np.cross(velocity[i], args.B))
        a[i] = a[i] + F / mass_of_argon + [0.0, 0.0, 9.8]
    return a

def update_pos(x, v, a, dt):
    return x + v * dt + 0.5 * a * dt * dt

def update_velo(v, a, a1, dt):
    return v + 0.5 * (a + a1) * dt

def run_md(dt, number_of_steps, initial_temp, xyz, args):
    v = init_velocity(initial_temp, len(xyz))
    a = get_accelerations(xyz, v, args)
    for _ in range(number_of_steps):
        xyz = update_pos(xyz, v, a, dt)
        a1 = get_accelerations(xyz, v, args)
        v = update_velo(v, a, a1, dt)
        a = np.array(a1)

        print(xyz[0][0], xyz[0][1], xyz[0][2])

def main(args):
    args.E = np.array([args.E[0], args.E[1], args.E[2]], dtype='float64')
    args.B = np.array([args.B[0], args.B[1], args.B[2]], dtype='float64')

    xyz = np.array([[0.0, 0.0, 0.0]])
    run_md(0.1, 10000, args.T, xyz, args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
