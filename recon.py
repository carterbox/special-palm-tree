import glob
import numpy as np
import tike.ptycho

def main()
    for file in glob.glob('data/*.npz'):
        archive = np.load(file)
        original = archive['original']
        data = archive['data']
        scan = archive['scan']
        code = archive['code']
        probe = archive['probe']
        nspot = np.sum(code)
        # Start with a guess of all zeros for psi
        result = {
            'psi': np.zeros(original.shape, dtype='complex64') + 1e-32,
            'probe': probe,
        }
        result = tike.ptycho.reconstruct(
            data=data,
            scan=scan,
            **result,
            algorithm='cgrad',
            num_iter=64,
            nspot=nspot,
        )
        np.savez(
            f'result/{hash(tuple(code.tolist()))}',
            **result,
        )

if __name__ == '__main__':
    main()
