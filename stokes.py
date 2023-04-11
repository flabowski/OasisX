
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from dolfin import Expression, DirichletBC, Mesh, XDMFFile, MeshValueCollection
from dolfin import cpp, grad, ds, inner, dx, div, dot, solve, lhs, rhs, project
from dolfin import Constant, Function, VectorElement, FiniteElement, plot
from dolfin import FunctionSpace, TestFunction, TrialFunction, split, assemble, VectorFunctionSpace# , MixedFunctionSpace

mesh_name = "./resources/freezing_cavity/mesh.xdmf"
facet_name = "./resources/freezing_cavity/mf.xdmf"
bc_dict = {"fluid": 0, "bottom": 1, "right": 2, "top": 3, "left": 4}

mesh = Mesh()
with XDMFFile(mesh_name) as infile:
    infile.read(mesh)
dim = mesh.topology().dim()
mvc = MeshValueCollection("size_t", mesh, dim - 1)
with XDMFFile(facet_name) as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
ds_ = ds(subdomain_data=mf, domain=mesh)
P2 = VectorElement("CG", mesh.ufl_cell(), 2)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
TH = P2 * P1
VQ = FunctionSpace(mesh, TH)
# V = VectorFunctionSpace(mesh, "CG", 2)
# Q = FunctionSpace(mesh, "CG", 1)
# VQ = MixedFunctionSpace([V, Q])
# mf = self.mf
no_slip = Constant((0.0, 0))
topflow = Constant((1.0, 0))
bc0 = DirichletBC(VQ.sub(0), topflow, mf, bc_dict["top"])
bc1 = DirichletBC(VQ.sub(0), no_slip, mf, bc_dict["left"])
bc2 = DirichletBC(VQ.sub(0), no_slip, mf, bc_dict["bottom"])
bc3 = DirichletBC(VQ.sub(0), no_slip, mf, bc_dict["right"])
bc4 = DirichletBC(VQ.sub(1), Constant(0), mf, bc_dict["top"])
bcs = [bc0, bc1, bc2, bc3, bc4]

vup = TestFunction(VQ)
up = TrialFunction(VQ)
up_ = Function(VQ)  # Function holding the solution

u, p = split(up)  # Trial
vu, vp = split(vup)  # Test

mu = 1/1000
x, y = VQ.tabulate_dof_coordinates().T
rho = Function(VQ).sub(0).sub(0)
rho.vector().vec().array = 1000 - x * 1. + y * .1

g = Constant((0.0, -9.81))
F = (
    - mu * inner(grad(vu), grad(u)) * dx
    + inner(div(vu), p) * dx
    + inner(vp, div(u)) * dx
    + dot(g*rho, vu) * dx
)
solve(lhs(F) == rhs(F), up_, bcs=bcs)
u_, p_ = up_.split()

def mesh2triang(mesh):
    import matplotlib.tri as tri
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

fig, ax =plt.subplots()
ax.tripcolor(mesh2triang(mesh), p_.vector().get_local())
plt.show()

# plot(u_, mesh=mesh)

uv = u_.vector().vec().array
# v = up_.sub(0).sub(1).vector().vec().array
p = up_.sub(1).vector().vec().array
# magnitude = (u ** 2 + v ** 2) ** 0.5

# # x, y = mesh.coordinates().T
# tri = mesh.cells()

# fig, ax = plt.subplots()
# c1 = ax.quiver(x, y, u, v, magnitude)
# #ax.tricontourf(x, y, tri, p, levels=40)
plt.show()
a =1