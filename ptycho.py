"""An experiment to test coded-exposure for deblurring fly-scans in ptychography."""

import matplotlib.pyplot as plt
import numpy as np
import tike.ptycho
from xdesign.codes import is_prime, mura_1d

for module in [tike, np]:
    print("{} is version {}".format(module.__name__, module.__version__))


def muras_less_than(L):
    """Returns the lengths of all MURAs with lengths less than L."""
    lengths = list()
    for m in np.arange(1, (L - 1) / 4):
        L1 = int(4 * m) + 1
        if is_prime(L1):
            lengths.append(L1)
    return lengths


def get_object():
    """Normalize input uint8 images to the `[0, 1]` and `[0, pi]` ranges."""
    phase = plt.imread(
        "/home/beams/DCHING/Pictures/images/coins_2048.tif", ) / 255 * np.pi
    phase = phase[650:650 + 258, :]
    amplitude = 1
    original = np.expand_dims(
        amplitude * np.exp(1j * phase),
        axis=0,
    ).astype('complex64')
    return original


def get_probe(pw=256):
    """The probe is an 8 pixel wide rectangle."""
    weights = np.zeros((pw, pw))
    weights[:, 0:8] = 1
    probe = weights * np.exp(1j * 0)
    probe = np.expand_dims(probe, axis=0).astype('complex64')
    return probe


def get_trajectory(code, pw, stride=4):
    right_edge = 2048 - pw
    left_edge = 0
    where_code = np.where(code)
    positions = list()
    for left in np.arange(left_edge, right_edge, stride):
        positions.append((left + where_code) % right_edge)
    positions = np.concatenate(positions, axis=1)
    v, h = np.meshgrid(0, positions, indexing='ij')
    scan = np.expand_dims(
        np.stack((np.ravel(v), np.ravel(h)), axis=1),
        axis=0,
    ).astype('float32')
    print(f"The shape of the scan is {scan.shape}")
    return scan

def flat_code(L):
    return np.ones(L, dtype=bool)

def simulate(noise=False):
    """nspot increases with the travel distance"""
    original = get_object()
    probe = get_probe()
    for L in 2**np.arange(3, 4):
        codes = [None, None]
        codes[0] = mura_1d(L)
        codes[1] = flat_code(len(codes[0]))
        for code in codes:
            scan = get_trajectory(code=code, pw=len(probe))
            nspot = np.sum(code)
            data = tike.ptycho.simulate(
                detector_shape=pw,
                probe=probe,
                scan=scan,
                psi=original,
                nspot=nspot,
            )
            if noise:
                np.random.seed(0)
                data = np.random.poisson(data)
            np.savez(
                f'data/{hash(tuple(code.tolist()))}',
                data=data,
                probe=probe,
                scan=scan,
                code=code,
                original=original,
            )

if __name__ == '__main__':
    simulate()
