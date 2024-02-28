""" Builds models used in the paper.

"""

from .loading import load_niklas_volcano_data

N_CHUNKS = 200


def build_fault_line_model(
    volcano_coords, kernel, lambda0, sigma0, coeff_cov="uniform", coeff_mean="uniform"
):
    # Volcano center.
    x0 = volcano_coords[:, 0].mean()
    y0 = volcano_coords[:, 1].mean()
    z0 = volcano_coords[:, 2].mean()

    origin_offset = 0
    theta = 90  # equatorial plane.
    phi = 135
    saturation_length = 2500

    dists = planar(
        volcano_coords,
        x0,
        y0,
        z0,
        phi,
        theta,
    )
    basis_fn = tanh_sigmoid(dists, saturation_length, inverted=True)
    coeff_F = torch.hstack(
        [torch.ones(volcano_coords.shape[0], 1), basis_fn.reshape(-1, 1)]
    ).float()

    model = UniversalUpdatableGP(
        kernel,
        lambda0,
        sigma0,
        volcano_coords,
        coeff_F,
        coeff_cov,
        coeff_mean,
        n_chunks=N_CHUNKS,
    )
    return model


def build_cylinder_model(volcano_coords, kernel, lambda0, sigma0):
    # Volcano center.
    x0 = volcano_coords[:, 0].mean()
    y0 = volcano_coords[:, 1].mean()

    dists = cylindrical(volcano_coords, x0, y0)
    basis_fn = tanh_sigmoid(dists**2, saturation_length=2e6, inverted=True)

    coeff_F = torch.hstack(
        [torch.ones(volcano_coords.shape[0], 1), basis_fn.reshape(-1, 1)]
    ).float()

    model = UniversalUpdatableGP(
        kernel,
        lambda0,
        sigma0,
        volcano_coords,
        coeff_F,
        coeff_cov="uniform",
        coeff_mean="uniform",
        n_chunks=200,
    )
    return model


def build_constant_model(volcano_coords, kernel, lambda0, sigma0):
    model = UniversalUpdatableGP(
        kernel,
        lambda0,
        sigma0,
        volcano_coords,
        torch.ones(volcano_coords.shape[0], 1).float(),
        coeff_cov="uniform",
        coeff_mean="uniform",
        n_chunks=200,
    )
    return model


def load_paper_models(base_folder):
    # Those are reasonable values used for init, they are trained afterwards.
    sigma0 = 284.66
    m0 = 2139.1
    lambda0 = 651.58

    niklas_volcano_data = load_niklas_volcano_data(base_folder)
    volcano_coords = niklas_volcano_data["volcano_coords"]

    return {
        "fault line": build_fault_line_model(volcano_coords, kernel, lambda0, sigma0),
        "cylinder": build_cylinder_model(volcano_coords, kernel, lambda0, sigma0),
        "constant": build_constant_model(volcano_coords, kernel, lambda0, sigma0),
    }
