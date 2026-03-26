"""
    DiscreteEL(KineticEnergy, PotentialEnergy, nodes, b, c, h)

Construct the discrete Euler-Lagrange map for a 3-stage Lobatto method.

Returns a `DEL(q0, qmid, q1, i)` function that evaluates the discrete EL equations
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

    function DEL(q0, qmid, q1, i)
        q = hcat(q0, qmid, q1)
        sum(a -> b[a] * KineticEnergy(Q(q, a), Qdot(q, a)) * dlag[i, a] +
                 h * b[a] * PotentialEnergy(Q(q, a), Qdot(q, a)) * lag[i, a],
            1:length(c))
    end

    return DEL
end


"""
    DELSolve(DEL, q0, qmid, q1; ftol=1e-14)

Solve for the next pair `(qmid_new, q1_new)` given `(q0, qmid, q1)`.
"""
function DELSolve(DEL, q0, qmid, q1; ftol=1e-14)
    d = length(q0)
    guess = [qmid + h/2 * qmid; q1 + h/2 * q1]
    function objective(Q)
        Qmid = @view Q[1:d]
        Qnext = @view Q[d+1:end]
        eq1 = DEL(q0, qmid, q1, 3) + DEL(q1, Qmid, Qnext, 1) -
              rho/6 * (-Qnext + 4*Qmid - 3*q1) -
              rho/6 * (3*q1 - 4*qmid + q0)
        eq2 = DEL(q1, Qmid, Qnext, 2) - 2*rho/3 * (Qnext - q1)
        return [eq1; eq2]
    end
    return nlsolve(objective, guess, autodiff=:forward, ftol=ftol)
end


"""
    DELSolve(DEL, q0, qmid, q1, steps; ftol=1e-14)

Compute a trajectory of `2*steps+1` configurations for the 3-stage method.
"""
function DELSolve(DEL, q0, qmid, q1, steps; ftol=1e-14)
    d = length(q0)
    trj = zeros(size(q0, 1), 2*steps + 1)
    trj[:, 1] = q0
    trj[:, 2] = qmid
    trj[:, 3] = q1
    @showprogress for j in 1:2:2*(steps-1)
        sol = DELSolve(DEL, @view(trj[:, j]), @view(trj[:, j+1]), @view(trj[:, j+2]), ftol=ftol).zero
        trj[:, j+3] = @view(sol[1:d])
        trj[:, j+4] = @view(sol[d+1:end])
    end
    return trj
end


function ConjugateMomenta_p0(DEL, q0, qmid, q1)
    return -DEL(q0, qmid, q1, 1) + rho/6 * (-q1 + 4*qmid - 3*q0)
end

function ConjugateMomenta_p1(DEL, q0, qmid, q1)
    return DEL(q0, qmid, q1, 3) - rho/6 * (3*q1 - 4*qmid + q0)
end


"""
    ConjugateMomenta(DEL, traj; steps)

Compute conjugate momenta along a trajectory matrix for the 3-stage method.
"""
function ConjugateMomenta(DEL, traj::Matrix; steps)
    p0 = [ConjugateMomenta_p0(DEL, @view(traj[:, 1]), @view(traj[:, 2]), @view(traj[:, 3]))]
    ps = [ConjugateMomenta_p1(DEL, @view(traj[:, i+1]), @view(traj[:, i+2]), @view(traj[:, i+3]))
          for i in range(0, step=2, length=steps)]
    return stack([p0; ps], dims=2)
end


"""
    Step1(DEL, q0, p0; ftol=1e-14)

Solve for the first step `(qmid, q1)` given initial conditions `(q0, p0)`.
"""
function Step1(DEL, q0, p0; ftol=1e-14)
    d = length(q0)
    guess = [q0 + h * p0 / 2; q0 + h * p0]
    function objective(Q)
        Qmid = @view Q[1:d]
        Qnext = @view Q[d+1:end]
        eq1 = p0 + DEL(q0, Qmid, Qnext, 1) -
              rho/6 * (-Qnext + 4*Qmid - 3*q0)
        eq2 = DEL(q0, Qmid, Qnext, 2) - 2*rho/3 * (Qnext - q0)
        return [eq1; eq2]
    end
    return nlsolve(objective, guess, autodiff=:forward, ftol=ftol)
end


"""
    ComputeTrajectory(DEL, q0, p0, steps; ftol=1e-14)

Compute a full trajectory starting from initial conditions `(q0, p0)` for the 3-stage method.
"""
function ComputeTrajectory(DEL, q0, p0, steps; ftol=1e-14)
    sol = Step1(DEL, q0, p0).zero
    d = length(q0)
    qmid = sol[1:d]
    q1 = sol[d+1:end]
    return DELSolve(DEL, q0, qmid, q1, steps, ftol=ftol)
end
