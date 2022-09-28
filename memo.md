# MEMO

## 3/18
Juho Lee님이 모두에게 11:33 AM
sigma_1(X)*branch_1(X) + sigma_2(X)*branch_2(X) + sigma_3(X)*branch_3(X)
sigma_1, sigma_2, sigma_3 = sigmoid
branch_1 = W_emlp_1, branch_2 = W_emlp_2
branch_3 = W_mlp
W_agg * concat[branch_1(X), branch_2(X), branch_3(X)]
W_agg(X) * concat[branch_1(X), branch_2(X), branch_3(X)]
Hongseok Yang님이 모두에게 11:49 AM
rho : G x X -> X
rho(g)(x) = x’
rho(g) = M_g
rho(g)(x) = M_g x
rho(g)(x) = if x is good then M_g x else x

Q(3)
[[ 0.57735026]
 [-0.        ]
 [-0.        ]
 [-0.        ]
 [ 0.57735026]
 [-0.        ]
 [-0.        ]
 [-0.        ]
 [ 0.57735026]]
Qxy
[[0.70710677 0.        ]
 [0.         0.        ]
 [0.         0.        ]
 [0.         0.        ]
 [0.70710677 0.        ]
 [0.         0.        ]
 [0.         0.        ]
 [0.         0.        ]
 [0.         1.        ]]
Qyz
[[0.70710677 0.        ]
 [0.         0.        ]
 [0.         0.        ]
 [0.         0.        ]
 [0.70710677 0.        ]
 [0.         0.        ]
 [0.         0.        ]
 [0.         0.        ]
 [0.         1.        ]]