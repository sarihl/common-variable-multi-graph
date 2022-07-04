import torch
import numpy as np
from warnings import warn

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CommonVariable:
    def __init__(self, laplacian_list, target_opt_error=1e-6, lr_rate=1e-2, no_imp_lr_red=10, stop_no_imp=30,
                 device=DEFAULT_DEVICE):
        self.no_imp_lr_red = no_imp_lr_red
        self.stop_no_imp = stop_no_imp
        self.lr_rate = lr_rate
        if target_opt_error < 0:
            warn(f'negative target optimization error {target_opt_error}, using the absolute value')
            self.opt_error = - target_opt_error
        elif target_opt_error == 0:
            warn(f'the target optimization error is 0, the algorithm will likely not terminate.')
            self.opt_error = target_opt_error
        else:
            self.opt_error = target_opt_error
        self.laplacian_list = torch.as_tensor(laplacian_list).to(device)
        assert len(self.laplacian_list.shape) == 3 and (self.laplacian_list.shape[-1] == self.laplacian_list.shape[-2])
        self.num_graphs = self.laplacian_list.shape[0]
        self.ts = torch.rand(self.num_graphs, device=device) + 1e-6
        self.ts /= torch.sum(self.ts)
        assert torch.isclose(self.ts.sum(), torch.as_tensor(1.0)) and torch.all((1 >= self.ts) * (self.ts >= 0))
        self.lap_dif_mat, self.lambda_1_list = self._calc_lapcian_dif_tensor()

        self.common_variable = None
        self.final_error_bound = None
        self._valid_output = False

    def _calc_lapcian_dif_tensor(self):
        """
        this function calculates the constant matrices that appear inside the expression for the gradient shown in
        Lemma 1.
        :return: a tuple of:
            1. a tensor of shape (mxnxn) where m is the number of graphs and n is the number of vertices.
            2. the second eigen value of each matrix
        note: the function assumes that the laplacians are symteric.
        """
        assert torch.allclose(self.laplacian_list.permute([0, 2, 1]), self.laplacian_list)
        eig_vals = torch.linalg.eigvalsh(self.laplacian_list)[:, 1]
        assert torch.all(eig_vals > 0)
        norm_lap_list = self.laplacian_list / eig_vals[:, None, None]
        res = norm_lap_list[:-1, ...] - norm_lap_list[-1, ...]
        return res, eig_vals

    def calc_common_smoothness(self, x):
        bilinear_out = _batch_bilinear_mul(x, self.laplacian_list, x)
        smoothness_list = bilinear_out / self.lambda_1_list
        assert len(smoothness_list.shape) == 1 and smoothness_list.shape[0] == self.num_graphs
        return torch.max(smoothness_list)

    def fit(self, x=None, y=None):
        with torch.no_grad():
            clip_count = 0
            no_imp_count = 1
            best_err_bound = np.inf
            while True:
                # calculate the averaged laplacian
                L_avg = torch.sum(self.laplacian_list / self.lambda_1_list[..., None, None] * self.ts[..., None, None],
                                  dim=0)

                # calculate eigen decomposition
                eigvals, eigvecs = torch.linalg.eigh(L_avg)
                lambda_1 = eigvals[1]
                phi_1 = eigvecs[:, 1]  # TODO: check this

                # calc optimization error bound
                smoothness = self.calc_common_smoothness(phi_1)
                err_bound = smoothness - lambda_1
                # stopping condition
                if err_bound <= self.opt_error:
                    break

                if err_bound < best_err_bound:
                    best_err_bound = err_bound
                    no_imp_count = 1
                else:
                    no_imp_count += 1
                if (self.no_imp_lr_red is not None) and no_imp_count % self.no_imp_lr_red == 0:
                    self.lr_rate /= 10
                if (self.stop_no_imp is not None) and no_imp_count % self.stop_no_imp == 0:
                    break

                # optimization
                grad = _batch_bilinear_mul(phi_1, self.lap_dif_mat, phi_1)
                assert len(grad.shape) == 1 and grad.shape[0] == (self.num_graphs - 1)
                ts_clipped = self.ts[:-1]
                ts_clipped += grad * self.lr_rate
                if (not torch.all((0 <= ts_clipped) * (ts_clipped <= 1))) or torch.sum(
                        ts_clipped) > 1:  # TODO: maybe do this dynamically?
                    clip_count += 1
                    ts_clipped[ts_clipped < 0] = 0
                    ts_clipped[ts_clipped > 1] = 1
                    if torch.sum(ts_clipped) > 1:
                        ts_clipped /= 1.1 * torch.sum(ts_clipped)
                    if clip_count > 5:
                        warn('t values have been clipped 5 iterations consecutively, lr decreased by 10 times.')
                        self.lr_rate /= 10
                        clip_count = 0
                else:
                    clip_count = 0
                self.ts[:-1] = ts_clipped
                self.ts[-1] = 1 - torch.sum(ts_clipped)
            self.final_error_bound = err_bound.detach().cpu().numpy()
            self.common_variable = phi_1.detach().cpu().numpy()
            self._valid_output = True

    def is_valid(self):
        return self._valid_output

    def get_optimal_laplacian(self):
        if not self._valid_output:
            raise ValueError('The fit function should be called first')
        L = L_avg = torch.sum(self.laplacian_list / self.lambda_1_list[..., None, None] * self.ts[..., None, None],
                              dim=0)
        return L.detach().cpu().numpy()


def _batch_bilinear_mul(a, A, b):  # TODO: test
    # preparing a
    if len(a.shape) == 1 or (len(a.shape) == 2 and (1 in a.shape)):
        a = a.squeeze()[None, :, None]
    elif len(a.shape) == 2:
        a = a[..., None]

    # preparing b
    if len(b.shape) == 1 or (len(b.shape) == 2 and (1 in b.shape)):
        b = b.squeeze()[None, None, :]
    elif len(b.shape) == 2:
        b = b[:, None, :]

    # preparing A
    if len(A.shape) == 2:
        A = A[None, ...]

    # calculation
    return torch.sum(a * A * b, dim=[-2, -1])


def _sanity_check():
    import numpy as np

    n = 10
    num_lap = 5

    L = np.eye(n) * 2
    L[np.arange(n - 1), np.arange(1, n)] = -1
    L[np.arange(1, n), np.arange(n - 1)] = -1
    L_list = np.zeros((num_lap, 1, 1)) + L[None, ...]
    alg = CommonVariable(L_list)
    alg.fit()
    common_var = alg.common_variable
    return common_var


if __name__ == '__main__':
    _sanity_check()
