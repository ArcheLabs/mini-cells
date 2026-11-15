# Mathematical specification

Memory computes x + LN(Wout GELU(Win x)). The token reconstruction
objective uses mean squared error in this implementation. For every
token, compute g at the reference weights and apply
S' = eta S − 0.05 rate g and W' = (1 − 0.01 forget) W + S'.
The four controller outputs use independent sigmoid coordinates.
Address covariance evolves as C' = rho C + (1 − rho) uuᵀ with unit key u.
Read with the updated state and solve (C' + ridge I) z = C'q before
applying the MLP. Ridge is strictly positive.

Chunk length one is the sequential reference. Longer chunks keep the
gradient reference fixed at the chunk boundary while advancing all four
controls and state transitions at each token. This implementation uses
a serial PyTorch scan. The soft projection and scan implement MiniCells
adaptations. A complete TNT comparison requires hierarchical training
and the second adaptation stage as specified in the TNT paper.
