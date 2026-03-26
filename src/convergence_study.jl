"""
    convergence_study(KineticEnergy, PotentialEnergy, nodes, b, c, q0, p0,
                      q_exact, p_exact; T=30, Ns=2 .^(5:12), ftol=1e-14)

Run a convergence study over a range of step counts `Ns` for a given
discrete Euler-Lagrange integrator.

# Arguments
- `KineticEnergy`, `PotentialEnergy`: energy functions from `Coupled_Oscillator`
- `nodes, b, c`: Butcher tableau data from `LobattoRKdata`
- `q0, p0`: initial conditions (vectors of length d)
- `q_exact, p_exact`: vectors of exact solution functions, e.g. `[q1, q2]` and `[p1, p2]`

# Keyword Arguments
- `T`: final integration time (default 30)
- `Ns`: vector of step counts (default `2 .^(5:12)`)
- `ftol`: Newton solver tolerance (default `1e-14`)

# Returns
`(hs, err_q, err_p)` — step sizes and max-norm errors in position and momentum.
"""
function convergence_study(KineticEnergy, PotentialEnergy, nodes, b, c, q0, p0,
                           q_exact, p_exact; T=30, Ns=2 .^(5:12), ftol=1e-14)
    d = length(q0)
    m = length(Ns)
    hs = T ./ Ns
    err_q = zeros(m)
    err_p = zeros(m)

    for i in 1:m
        N = Ns[i]
        h = hs[i]
        ts = (0:N) * h

        DEL = DiscreteEL(KineticEnergy, PotentialEnergy, nodes, b, c, h)
        q_num = ComputeTrajectory(DEL, q0, p0, N, ftol=ftol)
        p_num = ConjugateMomenta(DEL, q_num, steps=N)

        for j in 1:d
            err_q[i] = max(err_q[i], maximum(abs.(q_num[j, :] .- q_exact[j].(ts))))
            err_p[i] = max(err_p[i], maximum(abs.(p_num[j, :] .- p_exact[j].(ts))))
        end
    end

    return hs, err_q, err_p
end
