import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import pdb 
# Discrete time model of a quadcopter
Ad = sparse.csc_matrix([
  [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
  [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
  [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
  [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
  [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
  [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
  [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
  [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
  [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
])
Bd = sparse.csc_matrix([
  [0.,      -0.0726,  0.,     0.0726],
  [-0.0726,  0.,      0.0726, 0.    ],
  [-0.0152,  0.0152, -0.0152, 0.0152],
  [-0.,     -0.0006, -0.,     0.0006],
  [0.0006,   0.,     -0.0006, 0.0000],
  [0.0106,   0.0106,  0.0106, 0.0106],
  [0,       -1.4512,  0.,     1.4512],
  [-1.4512,  0.,      1.4512, 0.    ],
  [-0.3049,  0.3049, -0.3049, 0.3049],
  [-0.,     -0.0236,  0.,     0.0236],
  [0.0236,   0.,     -0.0236, 0.    ],
  [0.2107,   0.2107,  0.2107, 0.2107]])
[nx, nu] = Bd.shape

# Constraints
u0 = 10.5916
umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
umax = np.array([13., 13., 13., 13.]) - u0
xmin = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
                 -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
xmax = np.array([ np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf,
                  np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

# Objective function
Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
QN = Q
R = 0.1*sparse.eye(4)

# Initial and reference states
x0 = np.zeros(12)
xr = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

# Prediction horizon
N = 10

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective
# NAZA miss 2?
q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])

# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()
print(P.shape)
print(q.shape)
print(A.shape)
print(l.shape)
print(u.shape)
pdb.set_trace()
# Setup workspace
prob.setup(P, q, A, l, u)

# Simulate in closed loop
nsim = 15
import pdb 
for i in range(nsim):
    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')
    
    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    x0 = Ad@x0 + Bd@ctrl
    pdb.set_trace()
    print(f"i {i}, x0 {x0}")
    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)


# def mpc_control(A, B, horizon, z_init, obj_feature_ref, hand_pos_goal, num_hand=28, num_obj=8, xmin=None, xmax=None, umin=None, umax=None):
#     # ref: https://osqp.org/docs/examples/mpc.html
#     [dim_x, dim_u] = B.shape
#     N = horizon
#     num_ori_pos = 4
#     # Objective function
#     QN = np.zeros((dim_x,dim_x))  
#     for i in range(0,num_obj):
#         QN[2*num_hand + i, 2*num_hand + i] = 0 #1
#     for i in range(0,num_hand):
#         QN[i,i] = 1 if i < num_hand - num_ori_pos else 5
#     Q = np.zeros((dim_x,dim_x))  
#     for i in range(0,num_hand):
#         Q[i, i] = 1 if i < num_hand - num_ori_pos else 5
#     R = np.zeros((dim_u,dim_u))

#     # Q = sparse.eye(dim_x)
#     # QN = Q
#     # R = 0.1*sparse.eye(dim_u)

#     z_0 = z_init 
#     Z_ref = np.zeros((dim_x, 1)) 
#     Z_ref[0:num_hand, :] = hand_pos_goal[-1].reshape(-1, 1)
#     # Z_ref = np.zeros(((N+1)*dim_x, 1)) 
#     # Z_ref[N*dim_x+2*num_hand:N*dim_x+2*num_hand+num_obj,:] = obj_feature_ref.reshape(-1, 1)
#     # Z_ref[0:num_hand, :] = z_0[0:num_hand].reshape(-1, 1)
#     # for i in range(1,N+1):
#     #     Z_ref[i*dim_x:i*dim_x+num_hand, :] = hand_pos_goal[i-1].reshape(-1, 1)

#     # - quadratic objective
#     P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
#                         sparse.kron(sparse.eye(N), R)], format='csc')
#     # - linear objective
#     #pdb.set_trace()
#     q = np.hstack([(np.kron(np.ones(N), (-Q@Z_ref).squeeze())).squeeze(), (-QN@Z_ref).squeeze(), np.zeros(N*dim_u)])
#     #pdb.set_trace()
#     #q = np.hstack([np.kron(np.ones(N), -Q)@Z_ref[:-dim_x,:], -QN@Z_ref[-dim_x:,:], np.zeros(N*dim_u)])
    
#     # - linear dynamics
#     Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(dim_x)) + sparse.kron(sparse.eye(N+1, k=-1), A)
#     Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), B)
#     Aeq = sparse.hstack([Ax, Bu])
#     leq = np.hstack([-z_0, np.zeros(N*dim_x)])
#     ueq = leq
#     # - input and state constraints
#     Aineq = sparse.eye((N+1)*dim_x + N*dim_u)
#     lineq = np.hstack([np.kron(np.ones(N+1), np.array([-100]*dim_x)), np.kron(np.ones(N), umin)])
#     uineq = np.hstack([np.kron(np.ones(N+1), np.array([100]*dim_x)), np.kron(np.ones(N), umax)])
#     # - OSQP constraints
#     A = sparse.vstack([Aeq, Aineq], format='csc')
#     l = np.hstack([leq, lineq])
#     u = np.hstack([ueq, uineq])

#     # Create an OSQP object
#     prob = osqp.OSQP()
#     # Setup workspace
#     print(P.shape)
#     print(q.shape)
#     print(A.shape)
#     print(l.shape)
#     print(u.shape)
#     #pdb.set_trace()
#     prob.setup(P, q, A, l, u)
#     res = prob.solve()
#     if res.info.status == "solved":
#         #U_k = torch.tensor(res.x, dtype=torch.double).to(device)
#         u_k = res.x[-N*nu:] #-(N-1)*nu
#         #.reshape(N, dim_u) #U_k[:m].reshape(1, m)
#         return u_k
#     else:
#         print(f"OSQP did not solve the problem. Status: {res.info.status}")
#         pdb.set_trace()