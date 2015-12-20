"""Mesh representation and construction"""
from __future__ import division, print_function

import logging
import numpy as np
import numpy.linalg as la
from collections import namedtuple

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


# Second order elements.

from modepy import vandermonde

second_order_basis = vandermonde([
    lambda x: np.ones_like(x[0]),
    lambda x: x[0],
    lambda x: x[1],
    lambda x: x[0] * x[1],
    lambda x: x[0] ** 2,
    lambda x: x[1] ** 2 ],
    np.array([[-1, 0, 1, 0, -1, -1], [-1, -1, -1, 0, 1, 0]], dtype=np.float))

element_to_coeffs = la.inv(second_order_basis)

def declare(*vars):
    from pymbolic import var
    if len(vars) == 1:
        return var(vars[0])
    return (var(x) for x in vars)

x, y, xc, yc, t = declare("x", "y", "xc", "yc", "t")
pt = declare("pt")

def basis_func(c):
    return c[0] + c[1] * x + c[2] * y + c[3] * x * y + c[4] * x ** 2 + c[5] * y ** 2

def combined_basis_func(c):
    return c[0] + c[1] * pt[0] + c[2] * pt[1] + c[3] * pt[0] * pt[1] + c[4] * pt[0] ** 2 + c[5] * pt[1] ** 2

xval = combined_basis_func(xc)
yval = combined_basis_func(yc)

from pymbolic import diff
from pymbolic import compile

dv = compile(diff(xval, pt[0]) * diff(yval, pt[1]) \
             - diff(yval, pt[0]) * diff(xval, pt[1])
             , ["pt", "xc", "yc"])

trans = compile(np.array([combined_basis_func(xc), combined_basis_func(yc)]), ["pt", "xc", "yc"])

#jacobian = np.vectorize(jacobian, excluded=[1, 2])
#trans = np.vectorize(trans, excluded=[1, 2])

# Edge parameterizations
edge0_param = (2 * t - 1, -1)
edge1_param = (1 - 2 * t, 2 * t - 1)
edge2_param = (-1, 1 - 2 * t)


def make_edge(edge_param):
    from pymbolic import substitute
    edge_x = substitute(basis_func(xc), {x: edge_param[0], y: edge_param[1]})
    edge_y = substitute(basis_func(yc), {x: edge_param[0], y: edge_param[1]})
    return compile(np.array((edge_x, edge_y)), ["xc", "yc", "t"])


def make_unit_normal(edge_param):
    from pymbolic import substitute
    sqrt = declare("sqrt")

    edge_x = substitute(basis_func(xc), {x: edge_param[0], y: edge_param[1]})
    tangent_x = diff(edge_x, t)

    edge_y = substitute(basis_func(yc), {x: edge_param[0], y: edge_param[1]})
    tangent_y = diff(edge_y, t)

    tangent_len = sqrt(tangent_x ** 2 + tangent_y ** 2)
    normal = np.array((tangent_y / tangent_len, -tangent_x / tangent_len))
    normal = compile(normal, ["sqrt", "xc", "yc", "t"])

    from functools import partial
    return partial(normal, np.sqrt)


def make_arc_length_element(edge_param):
    from pymbolic import substitute
    sqrt = declare("sqrt")

    edge_x = substitute(basis_func(xc), {x: edge_param[0], y: edge_param[1]})
    tangent_x = diff(edge_x, t)

    edge_y = substitute(basis_func(yc), {x: edge_param[0], y: edge_param[1]})
    tangent_y = diff(edge_y, t)

    ds = sqrt(tangent_x ** 2 + tangent_y ** 2)
    ds = compile(ds, ["sqrt", "xc", "yc", "t"])

    from functools import partial
    return partial(ds, np.sqrt)


# Edge mapping functions

edge0 = make_edge(edge0_param)
edge1 = make_edge(edge1_param)
edge2 = make_edge(edge2_param)

normal0 = make_unit_normal(edge0_param)
normal1 = make_unit_normal(edge1_param)
normal2 = make_unit_normal(edge2_param)

ds0 = make_arc_length_element(edge0_param)
ds1 = make_arc_length_element(edge1_param)
ds2 = make_arc_length_element(edge2_param)


Mesh = namedtuple("Mesh", "elements, boundary")


class AbstractElement(object):

    def pt(self, pt):
        raise NotImplementedError()

    def ds(self, pt):
        raise NotImplementedError()


class BoundaryPanel(AbstractElement):

    def __init__(self, boundary_points, point_numbers):
        self.boundary_points = boundary_points
        self.point_numbers = point_numbers

    def pt(self, pt):
        return self.point_numbers[pt,]

    def pt_coords(self, pt):
        return np.rollaxis(np.array([self.boundary_points[p].pt for p in np.array(pt)]), 1)

    def ds(self, pt):
        return np.array([self.boundary_points[p].ds for p in np.array(pt)])

    def n(self, pt):
        return np.rollaxis(np.array([self.boundary_points[p].n for p in np.array(pt)]), 1)


class BoundaryPoint(object):

    def __init__(self, edge, t):
        self.edge = edge
        self.t = t

    @property
    def ds(self):
        return self.edge.ds(self.t)

    @property
    def pt(self):
        return self.edge.pt(self.t)

    @property
    def n(self):
        return self.edge.n(self.t)

    @property
    def expansion_radius(self):
        return self.edge.approx_expansion_radius

    def expansion_center(self, r):
        return self.pt + r * self.n


class QuadraticElement(AbstractElement):

    def __init__(self, number, vertices, vertex_coords):
        self.number = number
        self.vertices = vertices
        self.vertex_coords = vertex_coords
        xs = [coord[0] for coord in vertex_coords]
        ys = [coord[1] for coord in vertex_coords]
        self.xs_in_order = np.array([xs[0], xs[3], xs[1], xs[4], xs[2], xs[5]])
        self.ys_in_order = np.array([ys[0], ys[3], ys[1], ys[4], ys[2], ys[5]])
        self.xcoeffs = element_to_coeffs.dot(self.xs_in_order)
        self.ycoeffs = element_to_coeffs.dot(self.ys_in_order)

    @property
    def edges(self):
        v = self.vertices
        return [(v[i], v[i + 3], v[(i + 1) % 3]) for i in range(3)]

    def pt(self, pt):
        return trans(pt, self.xcoeffs, self.ycoeffs)

    def ds(self, pt):
        return dv(pt, self.xcoeffs, self.ycoeffs)


class QuadraticElementEdge(object):

    def __init__(self, element, edge_number):
        self.element = element
        self.edge_number = edge_number

    def ds(self, t):
        ds_func = globals()["ds" + str(self.edge_number)]
        return ds_func(self.element.xcoeffs, self.element.ycoeffs, t)

    def pt(self, t):
        edge_func = globals()["edge" + str(self.edge_number)]
        return edge_func(self.element.xcoeffs, self.element.ycoeffs, t)

    def n(self, t):
        normal_func = globals()["normal" + str(self.edge_number)]
        return normal_func(self.element.xcoeffs, self.element.ycoeffs, t)

    @property
    def approx_expansion_radius(self):
        i = self.edge_number
        index = (2 * i, 2 * i + 1, (2 * i + 2) % 6),
        xs = self.element.xs_in_order[index]
        ys = self.element.ys_in_order[index]
        return sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)) / 2


def build_mesh_from_gmsh_receiver(receiver):
    from meshpy.gmsh_reader import GmshTriangularElement, GmshIntervalElement, GmshPoint
    elem_count = 0
    elems = []
    bdry = []

    for elem, elem_type in zip(receiver.elements, receiver.element_types):
        if isinstance(elem_type, GmshIntervalElement):
            # Add the boundary edge.
            bdry.append(tuple(elem))

        elif isinstance(elem_type, GmshTriangularElement):
            # Add a triangular element.
            vertex_coords = [receiver.points[vertex] for vertex in elem]
            elems.append(QuadraticElement(elem_count, elem, vertex_coords))
            elem_count += 1

        elif not isinstance(elem_type, GmshPoint):
            log.warning("encountered unexpected element type: {}".format(
                type(elem_type).__name__))

    # Match boundary edges with the boundary.
    bdry_segments = []
    for bdry_edge in bdry:
        bdry_edge = tuple(sorted(bdry_edge))
        matched = False
        for elem in elems:
            for edge_index, edge in enumerate(elem.edges):
                if bdry_edge == tuple(sorted(edge)):
                    assert not matched
                    bdry_segments.append(QuadraticElementEdge(elem, edge_index))
                    matched = True

    # Ensure all the segments have been accounted for.
    assert len(bdry) == len(bdry_segments)

    # Sort boundary segments (TODO).
    return Mesh(elements=elems, boundary=bdry_segments)


def gen_gmsh_code(f, boundary_points):
    assert len(boundary_points) % 2 == 0

    num_pts = len(boundary_points)

    code = []
    _ = lambda line: code.append(line)

    # Declare points on the boundary.
    for pt_index, pt in enumerate(boundary_points):
        x, y = f(pt)
        _("Point({pt_index}) = {{{x}, {y}, 0}};".format(pt_index=pt_index, x=x, y=y))

    # Join points into splines.
    for spline_index in range(num_pts // 2 - 1):
        _("Spline({spline_index}) = {{{left}, {mid}, {right}}};".format(
            spline_index=spline_index,
            left=2 * spline_index,
            mid=2 * spline_index + 1,
            right=2 * spline_index + 2))
    _("Spline({last_index}) = {{{left}, {mid}, 0}};".format(
        last_index=num_pts // 2 - 1,
        left=num_pts-2,
        mid=num_pts-1))

    # Create the loop and the surface.
    _("Line Loop({index}) = {{{lines}}};".format(
        index=num_pts,
        lines=",".join(str(line) for line in range(num_pts // 2))))

    _("Plane Surface(0) = {{{loop}}};".format(loop=num_pts))

    return "\n".join(code)


def make_2d_mesh(f, boundary, **kwargs):
    from meshpy.gmsh_reader import generate_gmsh, LiteralSource, GmshMeshReceiverNumPy
    source = LiteralSource(gen_gmsh_code(f, boundary, **kwargs), "geo")
    receiver = GmshMeshReceiverNumPy()
    generate_gmsh(receiver, source, dimensions=2, order=2)
    return build_mesh_from_gmsh_receiver(receiver)


def ds_Peanut(t):
    from numpy import sin, cos, sqrt
    return 0.125*sqrt((28.0*cos(2.0*t)*sin(2.0*t) + 56.0*cos(2.0*t))*cos(0.25*(4.0*t + -3.14159265359))*sin(0.25*(4.0*t + -3.14159265359)) + (35.0*sin(2.0*t)**2 + 28.0*sin(2.0*t))*cos(0.25*(4.0*t + -3.14159265359))**2 + (-1)*55.0*sin(2.0*t)**2 + 36.0*sin(2.0*t) + 100.0)


def get_equispaced_points(ds, total_length, n):
    points = [0]
    l = total_length / n

    def find_next(left):
        from scipy.integrate import quadrature
        from scipy.optimize import brent
        def obj(t):
            return (quadrature(ds, left, t, tol=1e-3, maxiter=100)[0] - l) ** 2
        return brent(obj, brack=(left, 2 * np.pi))

    for panel in range(n):
        points.append(find_next(points[-1]))

    return np.array(points)


def peanut(t):
    from numpy import cos, sin, pi
    return np.array([(3/4)*cos(t-pi/4)*(1+sin(2*t)/2), (sin(t-pi/4)*(1+sin(2*t)/2))])


def get_peanut_mesh(bdry_panels):
    from meshing import make_2d_mesh
    pts = get_equispaced_points(ds_Peanut, 6.41923977063, bdry_panels)
    return make_2d_mesh(peanut, pts[:-1])


def get_circle_mesh(bdry_panels):
    f = lambda t: np.array([np.cos(t), np.sin(t)])
    boundary = np.linspace(0, 2 * np.pi, bdry_panels + 1)[:-1]
    from meshing import make_2d_mesh
    return make_2d_mesh(f, boundary)


def test_quadratic_element():
    """Tests construction of quadratic elements."""
    elem = QuadraticElement(0, [0, 1, 2, 3, 4, 5],
                            [[1, 1], [2, 1], [1, 2], [1.5, 1.1], [1.54, 1.75], [1.0, 1.5]])
    edges = [QuadraticElementEdge(elem, j) for j in range(3)]
    xs = []
    ys = []
    nxs = []
    nys = []
    for edge in edges:
        print(edge.approx_expansion_radius)
        for pt in np.linspace(0, 1, 32):
           x, y = edge.pt(pt)
           xs.append(x)
           ys.append(y)

    import matplotlib.pyplot as plt
    plt.xlim(0.5, 2.5)
    plt.ylim(0.5, 2.5)

    plt.plot(xs, ys, marker=".", linewidth=0)

    del xs[:]
    del ys[:]

    for edge in edges:
        for pt in np.linspace(0, 1, 32)[1:-1]:
           x, y = edge.pt(pt)
           xs.append(x)
           ys.append(y)

           nx, ny = edge.n(pt)
           nxs.append(0.05 * nx)
           nys.append(0.05 * ny)

    for i in range(len(xs)):
        plt.arrow(xs[i], ys[i], nxs[i], nys[i], fc="k", ec="k", head_width=0.01)

    plt.grid(True)
    plt.axes().set_aspect("equal", "datalim")
    plt.show()


if __name__ == "__main__":
    test_quadratic_element()
