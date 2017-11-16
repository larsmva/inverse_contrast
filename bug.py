from dolfin import *
from fenics_adjoint import *


# [-50,50]^2

class DiffCoeff(Expression):
    def eval(self, values, x):
        if x[0] < -45.0 or x[1] < -45 or x[0] > 45 or x[1] > 45:
            values[0] = self.Dc
        elif x[0] < -40.0 or x[1] < -40 or x[0] > 40 or x[1] > 40:
            values[0] = self.Dg
        else:
            values[0] = self.Dw


class DerivDcDiffCoeff(Expression):
    def eval(self, values, x):
        if x[0] < -45.0 or x[1] < -45 or x[0] > 45 or x[1] > 45:
            values[0] = 1.0
        elif x[0] < -40.0 or x[1] < -40 or x[0] > 40 or x[1] > 40:
            values[0] = 0.0
        else:
            values[0] = 0.0


class DerivDgDiffCoeff(Expression):
    def eval(self, values, x):
        if x[0] < -45.0 or x[1] < -45 or x[0] > 45 or x[1] > 45:
            values[0] = 0.0
        elif x[0] < -40.0 or x[1] < -40 or x[0] > 40 or x[1] > 40:
            values[0] = 1.0
        else:
            values[0] = 0.0


class DerivDwDiffCoeff(Expression):
    def eval(self, values, x):
        if x[0] < -45.0 or x[1] < -45 or x[0] > 45 or x[1] > 45:
            values[0] = 0.0
        elif x[0] < -40.0 or x[1] < -40 or x[0] > 40 or x[1] > 40:
            values[0] = 0.0
        else:
            values[0] = 1.0


class ContrastForamenMagnum(Expression):
    def eval(self, values, x):
        if near(x[1], -50) and (x[0] > -5 and x[0] < 5):
            values[0] = 1
        else:
            values[0] = 0


def boundary(x, on_boundary):
    if on_boundary and near(x[1], -50) and (x[0] > -5 and x[0] < 5):
        return True
    else:
        return False


N = 50
mesh = UnitSquareMesh(N, N)
# use dimention: mm, hours
mesh.coordinates()[:] -= 0.5
mesh.coordinates()[:] *= 100
V = FunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)

U = Function(V)
U_prev = Function(V)

T = 2.0
t = 0
dt_val = 0.2
dt = Constant(dt_val)
# 20 time steps error
D = DiffCoeff(degree=1)
D.Dc = Constant(1000)
D.Dg = Constant(1)
D.Dw = Constant(2)

D.user_defined_derivatives = {D.Dc: DerivDcDiffCoeff(degree=1),
                              D.Dg: DerivDgDiffCoeff(degree=1),
                              D.Dw: DerivDwDiffCoeff(degree=1), }
ctrls = [Control(D.Dc), Control(D.Dg), Control(D.Dw)]
D_proj = project(D, V)
# plot(D_proj)

c_proj = project(ContrastForamenMagnum(degree=1), V)
# plot(c_proj)

a = u * v * dx + dt * D * inner(grad(u), grad(v)) * dx
L = U_prev * v * dx
# A = assemble(a)
bc = DirichletBC(V, ContrastForamenMagnum(degree=1), boundary)

uFile = File("U.pvd")
bc.apply(U.vector())
uFile << U

write_observations = False
if write_observations:
    observations = HDF5File(mpi_comm_world(), "U.xdmf", "w")
else:
    observations = HDF5File(mpi_comm_world(), "U.xdmf", "r")
    obs_func = Function(V)

J = 0
while t <= T:

    solve(a == L, U, bc)

    U_prev.assign(U)

    # plot(U)
    uFile << U

    t += dt_val
    print("Time ", t)

    # Write observation
    if write_observations:
        observations.write(U, str(t))
    else:
        try : # Only read some observations
            observations.read(obs_func, str(t))
            J += assemble((U - obs_func) ** 2 * dx)
        except :
            print "Error"
if write_observations:
    exit()

print("Functional value:", J)

print("Computing gradient")

Jhat = ReducedFunctional(J, ctrls)

# exit()
m = [Constant(1000), Constant(2), Constant(2)]
U.vector()[:] = 0
U_prev.vector()[:] = 0

for i in range(10):
    # Evaluate forward model at new control values
    j = Jhat(m)
    print("Functional value at iteration {}: {}".format(i, j))

    # Compute gradient
    dJdm = compute_gradient(J, ctrls)

    # Update control values:
    alpha = 0.1
    m = [Constant(m[0] - alpha * dJdm[0]),
         Constant(m[1] - alpha * dJdm[1]),
         Constant(m[2] - alpha * dJdm[2])]
    print([float(mm) for mm in m])

#exit()
print("Running Taylor test")

from IPython import embed;

embed()
h = [Constant(10), Constant(1), Constant(1)]
taylor_test_multiple(Jhat, m, h)
