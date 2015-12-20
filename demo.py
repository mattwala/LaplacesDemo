from __future__ import division, print_function

import numpy as np
import logging
from scipy.special import jn, jn_zeros

log = logging.getLogger(__name__)

log.setLevel(logging.WARNING)


alpha = jn_zeros(1, 3)[-1]


def f(multiplier, pt):
    return multiplier * jn(1, alpha * np.sqrt(pt[0] ** 2 + pt[1] ** 2)) * np.cos(np.arctan2(pt[1], pt[0]))


def plot_solution(targets, zs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    xs, ys = zip(*targets)

    from matplotlib import cm
    ax.plot_trisurf(xs, ys, zs, cmap=cm.jet, linewidth=0.01)
    plt.show()


def make_panels_into_targets(panels):
    points = []
    for panel in panels:
        points.extend([bp.pt for bp in panel.boundary_points])
    return points


def split_targets_into_far_and_close(targets, boundary_panels):
    points_per_panel = len(boundary_panels[0].boundary_points)
    sources = np.rollaxis(np.array(make_panels_into_targets(boundary_panels)), 1)
    far_targets = []
    close_targets = []
    expn_centers = []

    splitting = np.zeros(len(targets), dtype=bool)

    # QBX threshold. The QBX kernel is used for points within this distance to
    # the boundary.
    threshold = 0.1

    for i, target in enumerate(targets):
        distsq = (target[0] - sources[0]) ** 2 + (target[1] - sources[1]) ** 2
        min_dist_index = np.argmin(distsq)

        if distsq[min_dist_index] > threshold ** 2:
            splitting[i] = True
            continue

        # Found a close target.
        panel_index = min_dist_index // points_per_panel
        pt_index = min_dist_index % points_per_panel
        pt = boundary_panels[panel_index].boundary_points[pt_index]

        expn_centers.append(pt.expansion_center(-pt.expansion_radius))

    return (splitting, expn_centers)


class BoundaryFunction(object):

    def __init__(self, arr):
        self.arr = arr

    def __call__(self, pt):
        return self.arr[pt,]


def run_eoc(mesh, epss):
    errors = []
    from quadrature import MulticorePotentialEvaluator

    with MulticorePotentialEvaluator() as ev:

        from quadrature import NewtonKernel, ExactDoubleLayerKernel, QBXDoubleLayerKernel
        from quadrature import get_nystrom_matrix

        ev_mesh = mesh

        from modepy import VioreanuRokhlinSimplexQuadrature as VRSQ
        quad = VRSQ(16, 2)
        ev_quad = VRSQ(5, 2)

        exact_dbl_layer = ExactDoubleLayerKernel()
        qbx_dbl_layer = QBXDoubleLayerKernel(4)

        nystrom, nystrom_quad_rule, boundary_panels = get_nystrom_matrix(qbx_dbl_layer, mesh.boundary)
        bdry_targets = make_panels_into_targets(boundary_panels)

        targets = []
        weights = []
        for element in ev_mesh.elements:
            for i, pt in enumerate(np.rollaxis(ev_quad.nodes, 1)):
                targets.append(element.pt(pt))
                weights.append(ev_quad.weights[i] * element.ds(pt))

        targets = np.array(targets)
        far, expn_centers = split_targets_into_far_and_close(targets, boundary_panels)

        for eps in epss:
            newton = NewtonKernel(order=2, eps=eps)

            from functools import partial
            ff = partial(f, -alpha ** 2)

            bdry_pot_values = ev.eval(ff, newton, bdry_targets, quad, mesh.elements)
            bdry_condition = partial(f, 1)(np.rollaxis(np.array(bdry_targets), 1))

            from scipy.sparse.linalg import gmres
            rhs = bdry_condition - bdry_pot_values
            density = gmres(nystrom, rhs)[0]

            log.info("evaluating at {} targets ({} far, {} close)".format(
                len(targets), len(targets[far,]), len(targets[~far,])))

            density = BoundaryFunction(density)
            zs = np.empty(len(targets))
            zs[far,] = ev.eval(density, exact_dbl_layer, targets[far,], nystrom_quad_rule, boundary_panels)
            zs[~far,] = ev.eval(density, qbx_dbl_layer,
                                zip(expn_centers, targets[~far,]), nystrom_quad_rule, boundary_panels)

            #plot_solution(targets, zs)

            newton_zs = ev.eval(ff, newton, targets, quad, mesh.elements)

            #plot_solution(targets, newton_zs)

            zs += newton_zs

            #plot_solution(targets, zs)

            error = partial(f, 1)(np.rollaxis(targets, 1)) - zs
            error = np.sqrt(np.array(weights).dot(error ** 2))
            errors.append(error)

    print("epsilon\tl2 error")
    print("-----------------")
    for eps, error in zip(epss, errors):
        print("{}\t{}".format(eps, error))
    rate = np.polyfit([np.log(eps) for eps in epss], np.log(errors), 1)[0]
    print("EOC: " + str(rate))


if __name__ == "__main__":
    from meshing import get_circle_mesh, get_peanut_mesh

    print("Disk domain")
    run_eoc(get_circle_mesh(32), [0.1, 0.05, 0.01, .005, .001])

    print("Peanut domain")
    run_eoc(get_peanut_mesh(32), [0.1, 0.05, 0.01, .005, .001])
