# Fractional Variational Integrator using RKCQ (Algorithm 2)
#
# Usage (matching Coupled_Osci_ pattern — set globals, then call):
#
#   include("RKCQ_weights.jl")
#   W, c = convw_rk(N, T, RK, Ks)
#
#   nodes = c;  b = [1/6, 2/3, 1/6]   # Lobatto IIIC tableau
#   α = 0.5;  h = T / N
#   s = length(nodes);  m = s - 1
#
#   include("DEL_tools.jl")
#
#   f(t) = ...                          # forcing function
#   DL = DiscreteEL(f, nodes, b, c, h, N)
#   q  = ComputeTrajectory(DL, q₀, p₀, N)
