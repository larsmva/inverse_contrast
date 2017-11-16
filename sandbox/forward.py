

from dolfin import *

# [-50,50]^2
 
class DiffCoeff(Expression): 
  def eval(self, values, x):  
    if x[0] < -45.0 or x[1] < -45 or x[0] > 45 or x[1] > 45: values[0] = self.Dc  
    elif x[0] < -40.0 or x[1] < -40 or x[0] > 40 or x[1] > 40: values[0] = self.Dg  
    else: values[0] = self.Dw  

class ContrastForamenMagnum(Expression): 
  def eval(self, values, x):  
    if near(x[1],-50) and (x[0] > -5 and x[0] < 5): values[0] = 1   
    else: values[0] = 0   


def boundary(x, on_boundary): 
  if on_boundary and near(x[1],-50) and (x[0] > -5 and x[0] < 5): return True
  else: return False    
    



N= 100 
mesh = UnitSquareMesh(N,N)
# use dimention: mm, hours 
mesh.coordinates()[:] -= 0.5
mesh.coordinates()[:] *= 100  
V = FunctionSpace(mesh, "Lagrange", 1) 
u = TrialFunction(V)
v = TestFunction(V)

U = Function(V) 
U_prev = Function(V)

T = 10 
t= 0 
dt_val = 0.1
dt = Constant(dt_val) 

D = DiffCoeff(degree=1)
D.Dw = 2  
D.Dg = 1   
D.Dc = 1000 
D_proj = project(D, V) 
plot(D_proj)

c_proj = project(ContrastForamenMagnum(degree=1), V) 
plot(c_proj)

a = u*v*dx + dt*D*inner(grad(u), grad(v))*dx 
L = U_prev*v*dx 
A = assemble(a) 
bc = DirichletBC(V, ContrastForamenMagnum(degree=1), boundary) 

uFile = File("U.pvd") 
bc.apply(U.vector())
uFile << U

while t<=T: 
  
  b = assemble(L) 
  bc.apply(b) 
  bc.apply(A) 
  solve(A, U.vector(), b)

  U_prev.assign(U) 

  plot(U)
  uFile << U

  t += dt_val 
  print "Time ", t 



