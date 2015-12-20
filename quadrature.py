# -*- coding: utf-8 -*-

"""Quadrature and kernels"""
from __future__ import division, print_function

import logging
import numpy as np
import numpy.linalg as la
from modepy import vandermonde

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


from collections import namedtuple


NystromQuadRule = namedtuple("NystromQuadRule", "nodes, weights")


def get_nystrom_matrix(kernel, boundary, order=10):
    from scipy.special import legendre
    legendre_data = legendre(order).weights
    quad_points, quad_weights = legendre_data[:,0], legendre_data[:,1]

    # Shift the quadrature points and weights so they are over the interval (0, 1).
    quad_weights *= 1/2
    quad_points = 1/2 + (1/2) * quad_points

    from meshing import BoundaryPoint, BoundaryPanel

    boundary_points = []
    boundary_panels = []
    for seg_number, segment in enumerate(boundary):
        panel_points = [BoundaryPoint(segment, pt) for pt in quad_points]
        n = len(quad_points)
        boundary_panel = BoundaryPanel(panel_points, (seg_number * n) + np.arange(0, n))
        boundary_points.extend(panel_points)
        boundary_panels.append(boundary_panel)

    log.info("starting to build Nyström matrix")
    from multiprocessing import Pool
    from functools import partial
    make_row = partial(build_matrix_row, kernel, boundary_points, quad_weights)
    p = Pool()
    rows = p.map(make_row, range(len(boundary_points)))
    p.close()
    p.join()
    log.info("done building Nyström matrix")

    matrix = np.vstack(rows) - (1/2) * np.eye(len(boundary_points))
    return (matrix, NystromQuadRule(nodes=np.arange(len(quad_points)), weights=quad_weights), boundary_panels)


def build_matrix_row(kernel, boundary_points, quad_weights, row_num):
    row = np.empty(len(boundary_points))
    target = boundary_points[row_num]
    radius = target.expansion_radius
    icenter = target.expansion_center(-radius)
    ecenter = target.expansion_center(radius)
    for j, source in enumerate(boundary_points):
        weight = quad_weights[j % len(quad_weights)]
        row[j] = (1/2) * (kernel(source.pt, source.n, icenter, target.pt)  + \
                          kernel(source.pt, source.n, ecenter, target.pt)) * \
                 source.ds * weight
    return row


class MulticorePotentialEvaluator(object):

    def __enter__(self):
        from multiprocessing import Pool
        self.pool = Pool()
        return self

    def __exit__(self, *args):
        self.pool.close()
        self.pool.join()

    def eval(self, f, kernel, targets, quad, source_elements):
        from functools import partial
        map_func = partial(potential_over_element, f, kernel, targets, quad)
        #return sum(map(map_func, source_elements))
        return sum(self.pool.map(map_func, source_elements))


def potential_over_element(f, kernel, targets, quad, elem):
    result = np.empty(len(targets))
    f_values = f(elem.pt(quad.nodes))
    ds = elem.ds(quad.nodes)
    src_kernel = kernel.bind_to_source(elem, quad.nodes)

    for i, target in enumerate(targets):
        result[i] = np.sum(quad.weights * f_values * ds * src_kernel(target))

    return result


class Kernel(object):

    def bind_to_source(self, element, source_points):
        raise NotImplementedError()


class NewtonKernel(Kernel):

    def __init__(self, order, eps):
        self.kernel = get_expanded_newton_kernel(order)
        self.eps = eps

    def bind_to_source(self, element, source_points):
        pt_values = element.pt(source_points)
        from functools import partial
        return partial(self.kernel, self.eps, pt_values)


class ExactDoubleLayerKernel(Kernel):

    def __init__(self):
        self.kernel = get_double_layer_kernel()

    def bind_to_source(self, element, source_points):
        sources = element.pt_coords(source_points)
        n_values = element.n(source_points)
        from functools import partial
        return partial(self.kernel, sources, n_values)


class QBXDoubleLayerKernel(Kernel):

    def __init__(self, order):
        self.kernel = get_expanded_double_layer_kernel(order)

    def bind_to_source(self, element, source_points):
        sources = element.pt_coords(source_points)
        n_values = element.n(source_points)
        from functools import partial
        bound_kernel = partial(self.kernel, sources, n_values)
        def kernel(target):
            # Target is a pair (center, target).
            return bound_kernel(target[0], target[1])
        return kernel

    def __call__(self, *args):
        return self.kernel(*args)


class MaximaSession(object):

    def __enter__(self):
        from subprocess import Popen, PIPE
        log.debug("establishing maxima session")
        self.maxima = Popen(["maxima", "--very-quiet", "--disable-readline"],
                            stdin=PIPE, stdout=PIPE, universal_newlines=True)
        self("load(f90)$")
        self("load(vect)$")
        self("ratprint:false$")

        return self

    def __call__(self, input_str):
        log.debug("maxima: {}".format(input_str))
        self.maxima.stdin.write(input_str + "\n")
        self.maxima.stdin.flush()
        if input_str.endswith("$"):
            # Output suppressed
            return
        if input_str.startswith("f90"):
            lines = []
            next_line = self.maxima.stdout.readline()
            while not next_line.endswith("false"):
                lines.append(next_line)
                next_line = self.maxima.stdout.readline().strip()
            self.last_output = "".join(lines)
            return self.last_output

    @property
    def expr(self):
        expr = self.last_output
        expr = expr.replace("&", "").replace("\n", "").replace("%pi", "pi")
        from pymbolic import var, substitute, parse
        expr = substitute(parse(expr), {var("pi"): np.pi})
        return expr

    def __exit__(self, *args):
        self("quit()$")
        self.maxima.wait()


def xy_to_vector(symbol):
    """
    Replaces varx, vary with var[0], var[1].
    """
    from pymbolic import var
    assert isinstance(symbol, var)
    vect = var(symbol.name[:-1])
    if symbol.name.endswith("x"):
        return vect[0]
    else:
        assert symbol.name.endswith("y")
        return vect[1]


def get_double_layer_kernel():
    with MaximaSession() as _:
        _("scalefactors([tx,ty])$")
        _("k(tx,ty):=(1 / (4 * %pi)) * grad(log((sx - tx)**2 + (sy - ty)**2)) . [nx,ny]$")
        _("ev(express(k(tx,ty)), diff)$")
        _("f90(%);")

    from pymbolic.mapper.substitutor import SubstitutionMapper
    expr = SubstitutionMapper(xy_to_vector)(_.expr)

    from pymbolic import compile
    return compile(expr, ["s", "n", "t"])


def get_expanded_double_layer_kernel(order):
    with MaximaSession() as _:
        _("scalefactors([tx,ty])$")
        _("k(tx,ty):=(1 / (4 * %pi)) * grad(log((sx - tx)**2 + (sy - ty)**2)) . [nx,ny]$")
        _("ev(express(k(tx,ty)), diff)$")
        _("taylor(%, [tx,ty], [cx,cy], [{0},{0}])$".format(order))
        _("f90(%);")

    from pymbolic.mapper.substitutor import SubstitutionMapper
    expr = SubstitutionMapper(xy_to_vector)(_.expr)

    # t: expansion target
    # c: expansion center
    # s: source point
    # n: outward unit normal to s
    from pymbolic import compile
    return compile(expr, ["s", "n", "c", "t"])


def get_expanded_newton_kernel(order):
    with MaximaSession() as _:
        _("k(eps) := (1 / (4 * %pi)) * log(rsq + eps**2)$")
        _("f90(subst([c=eps], taylor(k(eps), eps, -c, {})));".format(order))
        #_("f90((1/2) * (k(radius) + subst([c=radius], taylor(k(radius), radius, -c, {}))));".format(order))
        #_("f90(taylor(k(e), e, -c, {}));".format(order))

    from pymbolic import substitute, var, compile
    t = var("t")
    s = var("s")
    sqrt = var("sqrt")
    expr = substitute(_.expr, {"rsq": (s[0] - t[0]) ** 2 + (s[1] - t[1]) ** 2})
    compiled_expr = compile(expr, ["log", "eps", "s", "t"])

    from functools import partial
    return partial(compiled_expr, np.log)


def test_unit_circle():
    f = lambda t: np.array([np.cos(t), np.sin(t)])
    fprime = lambda t: np.array([-np.sin(t), np.cos(t)])
    boundary = np.linspace(0, 2 * np.pi, 21)[:-1]
    from meshing import make_2d_mesh
    disk = make_2d_mesh(f, fprime, boundary)
    kernel = get_double_layer_kernel(2)
    mat, bdry_pts = build_double_layer_nystrom_matrix(kernel, disk.boundary, 16)
    vec = np.array([p.pt[0] for p in bdry_pts])
    import matplotlib.pyplot as plt
    plt.plot(range(len(vec)), np.array(vec), linewidth=0, marker=".")
    plt.plot(range(len(vec)), mat.dot(vec), linewidth=0, marker=".")
    plt.show()


if __name__ == "__main__":
    test_unit_circle()
