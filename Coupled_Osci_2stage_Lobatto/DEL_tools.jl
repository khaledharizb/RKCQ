"""
    DiscreteEL(KineticEnergy, PotentialEnergy, nodes, b, c, h)

Construct the discrete Euler-Lagrange map for a 2-stage Lobatto method.

Returns a `DEL(q0, q1, i)` function that evaluates the discrete EL equations
at stage index `i`.
"""
function DiscreteEL(KineticEnergy, PotentialEnergy, nodes, b, c, h)
    n = length(nodes)

    function Basis(j, t)
        y = one(t)
        for i in 1:n
            if i != j
                y *= (t - nodes[i]) / (nodes[j] - nodes[i])
            end
        end
        return y
    end

    function dBasis(j, t)
        ForwardDiff.derivative(t -> Basis(j, t), t)
    end

    lag = stack(map(j -> Basis.(j, c), 1:n), dims=1)
    dlag = stack(map(j -> dBasis.(j, c), 1:n), dims=1)

    function Q(q, i)
        sum(k -> @views(q[:, k]) * lag[k, i], 1:n)
    end

    function Qdot(q, i)
        sum(k -> @views(q[:, k]) * dlag[k, i], 1:n) / h
    end

    function DEL(q0, q1, i)
        q = hcat(q0, q1)
        sum(a -> b[a] * KineticEnergy(Q(q, a), Qdot(q, a)) * dlag[i, a] +
                 h * b[a] * PotentialEnergy(Q(q, a), Qdot(q, a)) * lag[i, a],
            1:length(c))
    end

    return DEL
end


"""
    DELSolve(DEL, q0, q1; ftol=1e-14)

Solve for the next configuration `q2` given `(q0, q1)` using the discrete EL equations.
"""
function DELSolve(DEL, q0, q1; ftol=1e-14)
    d = length(q0)
    guess = copy(q0)
    objective(Q) = DEL(q0, q1, 2) + DEL(q1, @view(Q[1:d]), 1) -
                   rho/2 * (@view(Q[1:d]) - q1) - rho/2 * (q1 - q0)
    return nlsolve(objective, guess, autodiff=:forward, method=:newton, ftol=ftol)
end


"""
    DELSolve(DEL, q0, q1, steps; ftol=1e-14)

Compute a trajectory of `steps+1` configurations starting from `(q0, q1)`.
"""
function DELSolve(DEL, q0, q1, steps; ftol=1e-14)
    d = length(q0)
    trj = zeros(size(q0, 1), steps + 1)
    trj[:, 1] = q0
    trj[:, 2] = q1
    @showprogress for j in 1:(steps-1)
        sol = DELSolve(DEL, @view(trj[:, j]), @view(trj[:, j+1]), ftol=ftol).zero
        trj[:, j+2] = @view(sol[1:d])
    end
    return trj
end


function ConjugateMomenta_p0(DEL, q0, q1)
    return -DEL(q0, q1, 1) + rho/2 * (q1 - q0)
end

function ConjugateMomenta_p1(DEL, q0, q1)
    return DEL(q0, q1, 2) - rho/2 * (q1 - q0)
end


"""
    ConjugateMomenta(DEL, traj; steps)

Compute conjugate momenta along a trajectory matrix.
"""
function ConjugateMomenta(DEL, traj::Matrix; steps)
    p0 = [ConjugateMomenta_p0(DEL, @view(traj[:, 1]), @view(traj[:, 2]))]
    ps = [ConjugateMomenta_p1(DEL, @view(traj[:, i]), @view(traj[:, i+1])) for i in 1:steps]
    return stack([p0; ps], dims=2)
end


"""
    Step1(DEL, q0, p0; ftol=1e-14)

Solve for the first step `q1` given initial position `q0` and momentum `p0`.
"""
function Step1(DEL, q0, p0; ftol=1e-14)
    d = length(q0)
    guess = q0 + h * p0
    objective(Q) = p0 + DEL(q0, @view(Q[1:d]), 1) - rho/2 * (@view(Q[1:d]) - q0)
    return nlsolve(objective, guess, autodiff=:forward, method=:newton, ftol=ftol)
end


"""
    ComputeTrajectory(DEL, q0, p0, steps; ftol=1e-14)

Compute a full trajectory starting from initial conditions `(q0, p0)`.
"""
function ComputeTrajectory(DEL, q0, p0, steps; ftol=1e-14)
    sol = Step1(DEL, q0, p0).zero
    d = length(q0)
    q1 = sol[1:d]
    return DELSolve(DEL, q0, q1, steps, ftol=ftol)
end
