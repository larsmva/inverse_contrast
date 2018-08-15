from dolfin import *
from fenics_adjoint import *

set_log_level(INFO)

# Import mesh and subdomains
mesh = Mesh("coarse_mesh.xml")
subdomains = MeshFunction("size_t", mesh, "coarse_sub_corrected.xml")

# Setup boundaries
load_bdy_from_file = True
D = mesh.topology().dim()
if load_bdy_from_file:
    # Load from file. This must be done for parallel.
    boundaries = MeshFunction("size_t", mesh, "coarse_bdy_corrected.xml")
else:
    # Mark boundaries and save to file. This must be done in serial.
    boundaries = MeshFunction("size_t", mesh, D - 1)
    mesh.init(D - 1, D)
    boundaries.set_all(0)
    for f in facets(mesh):
        if len(f.entities(D)) == 1:
            boundaries.array()[f.index()] = subdomains[int(f.entities(D))]
    File("coarse_bdy_corrected.xml") << boundaries

# Define measures with subdomains
dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Define function space
V = FunctionSpace(mesh, "CG", 1)


def forward_problem(D, g_list, tau, alpha=0.0, beta=0.0):
    """Compute the forward problem and return J + R.

    Args:
        D (dict or list): dict or list of diffusion coefficients,
            with keys/index being subdomain id corresponding to that coefficient.
        g_list (list): list of all boundary conditions.
            Length determines amount of timesteps
        tau (list): list of all observation time points.
            Last time point determines stop time (T) for simulation.
        alpha (float, optional): Regularisation parameter of boundary condition g
        beta (float, optional): Regularisation parameter of time derivative of g

    Returns:
        float: the resulting objective functional with regularisation: J + R.

    """
    # Define trial and test-functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Solution at current and previous time
    U = Function(V)
    U_prev = Function(V)

    # Hold index of next unused observation
    next_tau = 0

    # Set up the parameters
    k = len(g_list)  # Amount of time steps.
    T = tau[-1]
    t = 0
    dt = T / k

    # Open pipe to observations file
    obs_file = HDF5File(mpi_comm_world(), "U.xdmf", 'r')
    # Load initial observation as initial condition
    obs_file.read(U, "0")
    # Define observations function, which will load data from obs_file
    d = Function(V)

    # Define bilinear form, handling each subdomain 1, 2, and 3 in separate integrals.
    a = u * v * dx + sum([dt * D[j] * inner(grad(v), grad(u)) * dx(j) for j in range(1, 4)])
    # Define linear form.
    L = U_prev * v * dx

    # Impose Dirichlet boundary conditions on the boundary marked 1.
    g = g_list[0]
    bc = DirichletBC(V, g, boundaries, 1)
    current_g_index = 0

    # Assemble system matrix which is constant in time.
    A = assemble(a)
    # Apply boundary conditions to A
    bc.apply(A)

    # Define solver. Use GMRES iterative method with AMG preconditioner.
    solver = LinearSolver(mpi_comm_self(), "gmres", "amg")
    solver.set_operator(A)
    # solver.parameters["relative_tolerance"] = 1e-14

    # The functional
    J = 0
    import time
    while next_tau < len(tau):
        # Advance one timestep in time.
        U_prev.assign(U)  # Set newest U to previous U
        t += dt  # Increment time
        # Advance boundary condition function in time
        g_prev = g
        g = g_list[current_g_index]
        current_g_index += 1
        bc = DirichletBC(V, g, boundaries, 1)

        # Assemble RHS and apply DirichletBC
        b = assemble(L)
        A.bcs = [bc]
        bc.apply(b)

        # Solve linear system for this timestep
        s1 = time.clock()
        hmm = solver.solve(U.vector(), b)
        s2 = time.clock()
        print("hmm: ", hmm)
        print("Forward solve: ", s2 - s1)
        # list_timings(True, [TimingType_wall])

        if abs(t - tau[next_tau]) < abs(t + dt - tau[next_tau]):
            # If t is closest to next observation then compute misfit.
            obs_file.read(d, str(tau[next_tau]))  # Read observation
            J += assemble((U - d) ** 2 * dx)

            # Move on to next observation
            next_tau += 1

        # Choose time integral weights
        if t <= dt or next_tau >= len(tau):
            # If endpoints use 0.5 weight
            weight = 0.5
        else:
            # Otherwise 1.0 weight
            weight = 1.0

        # Add regularisation
        J += 1 / 2 * weight * dt * assemble(g ** 2 * ds(1)) * alpha
        if current_g_index > 1:
            J += 1 / 2 * weight * dt * assemble(((g - g_prev) / dt) ** 2 * ds(1)) * beta

    # We are done with reading observations
    obs_file.close()

    # Assert that all g are used.
    assert len(g_list[:current_g_index]) == len(g_list)

    return J

results_folder_load = "results-1e-2-1e-0-iter100-n2-tnc100"
results_folder_save = "results-1e-2-1e-0-iter100-n2-tnc500"

def save_control_values(m):
    h5file = HDF5File(mpi_comm_world(), "results/forward-problem/{}/opt_ctrls.xdmf".format(results_folder_save), 'w')
    if mpi_comm_self().rank == 0:
        myfile = open("results/forward-problem/{}/opt_consts.txt".format(results_folder_save), "w")
    for i, mi in enumerate(m):
        if isinstance(mi, Constant):
            c_list = mi.values()
            if mpi_comm_self().rank == 0:
                myfile.write("{}\n".format(str(float(c_list))))
        else:
            h5file.write(mi, str(i))
    h5file.close()
    if mpi_comm_self().rank == 0:
        myfile.close()


def load_control_values(k):
    m = []
    myfile = open("results/forward-problem/{}/opt_consts.txt".format(results_folder_load), "r")
    lines = myfile.readlines()
    for i in lines:
        m.append(Constant(float(i)))
    myfile.close()

    h5file = HDF5File(mpi_comm_world(), "results/forward-problem/{}/opt_ctrls.xdmf".format(results_folder_load), 'r')
    for i in range(k):
        mi = Function(V)
        h5file.read(mi, str(i + 3))
        m.append(mi)
    h5file.close()
    return m


if __name__ == "__main__":
    import numpy as np
    import numpy.random as rng

    rng.seed(22)
    D = {1: Constant(350), 2: Constant(0.8), 3: Constant(0.8)}
    k = 20
    g = [Function(V) for _ in range(k)]
    gfil = HDF5File(mpi_comm_world(), "g.xdmf", "r")
    bc = DirichletBC(V, 1.0, boundaries, 1)
    tmp_i_t = 0
    tmp_i_dt = 0.1
    for i, g_i in enumerate(g):
        tmp_i_t += tmp_i_dt
        g_i.vector()[:] = 0.0
    gfil.close()

    m = [D[1], D[2], D[3]] + g

    tau = [0.1, 0.30000000000000004, 0.7, 1.3,
           2.0000000000000004]
    alpha = AdjFloat(1E-2)
    beta = AdjFloat(1.0)

    J = forward_problem(D, g, tau, alpha=alpha, beta=beta)

    ctrls = ([Control(D[i]) for i in range(1, 4)]
             + [Control(g_i) for g_i in g])

    Jhat = ReducedFunctional(J, ctrls)
    Jhat.optimize()

    load_control_values_from_file = False
    if load_control_values_from_file:
        m = load_control_values(k)
        Jhat(m)

    lb = [1000 * 0.1, 1 * 0.1, 2 * 0.1]
    ub = [1000 * 10.0, 1 * 10.0, 2 * 10.0]

    for i in range(3, len(ctrls)):
        lb.append(0.0)
        ub.append(10.0)

    try:
        opt_ctrls = minimize(Jhat, method="L-BFGS-B", bounds=(lb, ub), callback = iter_cb, options={"disp": True, "maxiter": 100, "gtol": 1e-02})
    except RuntimeError as e:
        print(e)
        opt_ctrls = [itc.csf, itc.g, itc.w]
    print("End up: {} | {} | {}".format(float(opt_ctrls[0]), float(opt_ctrls[1]), float(opt_ctrls[2])))
    print(
    "[Constant({}), Constant({}), Constant({})]".format(float(opt_ctrls[0]), float(opt_ctrls[1]), float(opt_ctrls[2])))
    save_control_values(opt_ctrls)

