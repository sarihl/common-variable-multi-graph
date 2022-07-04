import numpy as np
from algorithm import CommonVariable
import matplotlib.pyplot as plt
import networkx as nx
import torch
from mpl_toolkits.mplot3d import Axes3D


def _closest_neighbor_bistochastic_laplacian(points: np.ndarray, num_neigh: int = 6):
    num_points = points.shape[1]
    norms = np.linalg.norm(points, axis=0)
    dist = norms[None, :] ** 2 + norms[:, None] ** 2 - 2 * points.T @ points
    ind = np.argsort(dist, axis=-1)
    ind = ind[:, 1:1 + num_neigh]
    L = np.zeros((num_points, num_points))
    L[np.arange(num_points).repeat(num_neigh), ind.ravel()] = -1
    L[L.T != 0] = -1
    assert np.allclose(L, L.T)
    diag = -np.sum(L, axis=-1)
    L[np.eye(num_points, dtype=bool)] = diag
    assert np.allclose(np.sum(L, axis=-1), 0) and np.allclose(np.sum(L, axis=-2), 0)
    L /= (np.sqrt(diag)[None, ...] * np.sqrt(diag)[..., None])
    return L


def fig_1_7(point_list, lap_list, axis_lim: float = 0.7, fig_num: int = 1):
    fig, axes = plt.subplots(1, len(point_list), figsize=(8 * len(point_list), 8))
    points = np.asarray(point_list)
    lap_list = lap_list.copy()
    for j in range(len(point_list)):
        lap = lap_list[j]
        lap[np.eye(lap.shape[0], dtype=bool)] = 0
        edge_list = np.argwhere(lap != 0)
        G = nx.Graph()
        G.add_edges_from(edge_list)
        # position is stored as node attribute data for random_geometric_graph
        # pos = nx.get_node_attributes(G, "pos")
        pos = {i: p for i, p in enumerate(points[j].T)}

        # find node near center (0.5,0.5)

        dist = np.linalg.norm(points[0], axis=-2)
        ncenter = np.argmin(dist).item()

        # color by path length from node near center
        p = dict(nx.single_source_shortest_path_length(G, ncenter))

        nx.draw_networkx_edges(G, pos, alpha=0.4, ax=axes[j])
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(p.keys()),
            node_size=80,
            node_color=list(p.values()),
            cmap=plt.cm.Reds_r,
            ax=axes[j]
        )

        axes[j].set_xlim(-axis_lim, axis_lim)
        axes[j].set_ylim(-axis_lim, axis_lim)
    plt.suptitle(f'Fig {fig_num}')
    plt.show()


def fig_2(lap_list: np.ndarray, num_points=100):
    eig_vals = np.linalg.eigvalsh(lap_list)[:, 1]
    t1 = np.linspace(0, 1, num_points)[:, None, None]
    Ls = lap_list[0][None, ...] * t1 / eig_vals[0] + lap_list[1][None, ...] * (1 - t1) / eig_vals[1]
    lambda_1 = np.linalg.eigvalsh(Ls)[:, 1]
    plt.plot(t1.squeeze(), lambda_1)
    plt.title('Fig 2')
    plt.ylabel('$\lambda_1(L_t)$')
    plt.xlabel('$t_1$')
    plt.show()


def fig_3(alg: CommonVariable, points: np.ndarray):
    assert alg.is_valid()
    common_var = alg.common_variable
    r = np.linalg.norm(points, axis=0)
    eig_vecs = torch.linalg.eigh(alg.laplacian_list)[1][:, :, 1]
    ind = np.argsort(r)
    eig_vecs = eig_vecs.cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(r[ind], common_var[ind])
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('$\psi_1(L_(t^{*}))$')
    axes[1].plot(r[ind], eig_vecs[0, ind])
    axes[1].set_xlabel('r')
    axes[1].set_ylabel('$\psi_1(L_1)$')
    axes[2].plot(r[ind], eig_vecs[1, ind])
    axes[2].set_xlabel('r')
    axes[2].set_ylabel('$\psi_1(L_2)$')
    fig.tight_layout(pad=3.0)
    fig.suptitle('Fig 3')
    plt.show()


def independent_2d_rotations(n=250):
    points = np.random.rand(2, n) - 1 / 2
    angles = np.random.rand(n) * 2 * np.pi
    rotation_matrix = np.asarray([[np.cos(angles), np.sin(angles)], [-np.sin(angles), np.cos(angles)]]).transpose(
        [-1, 0, 1])
    rotated_points = np.einsum('ijk,ki->ji', rotation_matrix, points)
    lap_list = np.stack(
        [_closest_neighbor_bistochastic_laplacian(points), _closest_neighbor_bistochastic_laplacian(rotated_points)],
        axis=0)

    fig_1_7([points, rotated_points], lap_list)
    fig_2(lap_list)
    alg = CommonVariable(lap_list)
    alg.fit()
    fig_3(alg, points)


def _sample_uniform_sphere(n=500, d=3):
    sample = []
    sampled = 0
    while sampled < n:
        num_points = np.int(np.ceil((n - sampled) * 8 / (4 * np.pi / 3)))
        points = np.random.rand(d, num_points) * 2 - 1
        r = np.linalg.norm(points, axis=0)
        points = points[:, r <= 1]
        sample.append(points)
        sampled += points.shape[1]
    sample = np.concatenate(sample, axis=-1)[:, :n]
    return sample


def network_plot_3D(G_list, angle=0, title=''):
    num_graphs = len(G_list)
    fig = plt.figure(figsize=plt.figaspect(0.5))

    for i, G in enumerate(G_list):
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')

        # Get number of nodes
        n = G.number_of_nodes()

        # Get the maximum number of edges adjacent to a single node
        edge_max = max([G.degree(i) for i in range(n)])

        # Define color range proportional to number of edges adjacent to a single node
        colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
        # 3D network plot
        ax = fig.add_subplot(1, num_graphs, i + 1, projection='3d')

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c=(colors[key],), s=40, edgecolors='k', alpha=0.7)
            ax.set_xlim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

        # Set the initial view
        ax.view_init(30, angle)

        # Hide the axes
        # ax.set_axis_off()
    plt.tight_layout(pad=1.0)
    plt.suptitle(title)
    plt.show()


def fig_4(point_list, lap_list):
    # prep graphs
    graphs = []
    for i, points_set in enumerate(point_list):
        lap = lap_list[i].copy()
        lap[np.eye(lap.shape[0], dtype=bool)] = 0
        edge_list = np.argwhere(lap != 0)
        G = nx.Graph()

        G.add_edges_from(edge_list)

        pos = {i: p for i, p in enumerate(points_set.T)}
        nx.set_node_attributes(G, pos, "pos")
        graphs.append(G)
    network_plot_3D(graphs, 0, 'Fig 4')


def fig_5_8(lap_list: np.ndarray, num_intervals: int = 50, fig_num: int = 5):  # TODO: fix
    eigs = np.linalg.eigvalsh(lap_list)[:, 1]
    t1, t2 = np.meshgrid(np.linspace(0, 1, num_intervals), np.linspace(0, 1, num_intervals))
    ts = np.stack([t1, t2, 1 - t1 - t2], axis=0)
    Lt = np.sum(lap_list[:, None, None, :, :] * ts[:, :, :, None, None] / eigs[:, None, None, None, None], axis=0)
    mask = t1 + t2 >= 1
    Lt[mask, ...] = 0
    lambda_1 = np.linalg.eigvalsh(Lt)[:, :, 1]
    cp = plt.contour(t1, t2, lambda_1)
    plt.clabel(cp)
    plt.title(f'Fig {fig_num}')
    plt.show()


def fig_6(alg: CommonVariable, points: np.ndarray):
    assert alg.is_valid()
    common_var = alg.common_variable
    r = np.linalg.norm(points, axis=0)
    eig_vecs = torch.linalg.eigh(alg.laplacian_list)[1][:, :, 1]
    ind = np.argsort(r)
    eig_vecs = eig_vecs.cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].plot(r[ind], common_var[ind])
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('$\psi_1(L_(t^{*}))$')
    axes[1].plot(r[ind], eig_vecs[0, ind])
    axes[1].set_xlabel('r')
    axes[1].set_ylabel('$\psi_1(L_1)$')
    axes[2].plot(r[ind], eig_vecs[1, ind])
    axes[2].set_xlabel('r')
    axes[2].set_ylabel('$\psi_1(L_2)$')
    axes[3].plot(r[ind], eig_vecs[2, ind])
    axes[3].set_xlabel('r')
    axes[3].set_ylabel('$\psi_1(L_3)$')
    fig.tight_layout(pad=3.0)
    fig.suptitle('Fig 6')
    plt.show()


def independent_3d_rotations(n=500):
    points = _sample_uniform_sphere(n, d=3)
    angles = np.random.rand(n) * 2 * np.pi
    rotation_matrix = np.asarray([[np.cos(angles), np.sin(angles)], [-np.sin(angles), np.cos(angles)]]).transpose(
        [-1, 0, 1])
    z_rotated_points = points.copy()
    z_rotated_points[:2, :] = np.einsum('ijk,ki->ji', rotation_matrix, points[:2, :])

    y_rotated_points = points[[0, 2, 1], :].copy()
    y_rotated_points[:2, :] = np.einsum('ijk,ki->ji', rotation_matrix, y_rotated_points[:2, :])
    y_rotated_points = y_rotated_points[[0, 2, 1], :]

    assert np.allclose(np.linalg.norm(points, axis=0), np.linalg.norm(y_rotated_points, axis=0))
    assert np.allclose(np.linalg.norm(points, axis=0), np.linalg.norm(z_rotated_points, axis=0))
    point_list = [points, z_rotated_points, y_rotated_points]
    lap_list = np.stack([_closest_neighbor_bistochastic_laplacian(p) for p in point_list], axis=0)
    fig_4(point_list, lap_list)
    fig_5_8(lap_list, fig_num=5)

    alg = CommonVariable(lap_list, target_opt_error=1e-8)
    alg.fit()
    fig_6(alg, points)


def fig_9(alg: CommonVariable, points: np.ndarray):
    Lt = alg.get_optimal_laplacian()
    embedding = np.linalg.eigh(Lt)[1][:, 1:4]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    NE_mask = (points[0] >= 0) * (points[1] >= 0)
    NW_mask = (points[0] < 0) * (points[1] >= 0)
    SW_mask = (points[0] >= 0) * (points[1] < 0)
    SE_mask = (points[0] < 0) * (points[1] < 0)
    ax.scatter(embedding[NE_mask, 0], embedding[NE_mask, 1], embedding[NE_mask, 2], marker="*", label='NE')
    ax.scatter(embedding[NW_mask, 0], embedding[NW_mask, 1], embedding[NW_mask, 2], marker="s", label='NW')
    ax.scatter(embedding[SW_mask, 0], embedding[SW_mask, 1], embedding[SW_mask, 2], marker="o", label='SW')
    ax.scatter(embedding[SE_mask, 0], embedding[SE_mask, 1], embedding[SE_mask, 2], marker="d", label='SE')
    plt.title('Fig 9')
    plt.legend()
    plt.show()


def barbel_experiment():
    points_1 = _sample_uniform_sphere(250, 2)
    points_2 = np.stack([points_1[0], points_1[1] * (1 - np.cos(np.pi * points_1[0]))], axis=0)
    points_3 = np.stack([points_1[0] * (1 - np.cos(np.pi * points_1[1])), points_1[1]], axis=0)

    point_list = [points_1, points_2, points_3]
    lap_list = np.stack([_closest_neighbor_bistochastic_laplacian(p) for p in point_list], axis=0)

    fig_1_7(point_list, lap_list, axis_lim=1.1, fig_num=7)
    fig_5_8(lap_list, fig_num=8)

    alg = CommonVariable(lap_list, lr_rate=1e-3)
    alg.fit()
    fig_9(alg, points_1)
