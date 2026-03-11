"""
    DiscreteEL(KineticEnergy, PotentialEnergy, nodes, b, c, h)

Construct the discrete Euler-Lagrange map for a 4-stage Lobatto method.

Returns a `DEL(q0, q01, q02, q1, i)` function that evaluates the discrete EL equations
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

    function DEL(q0, q01, q02, q1, i)
        q = hcat(q0, q01, q02, q1)
        sum(a -> b[a] * KineticEnergy(Q(q, a), Qdot(q, a)) * dlag[i, a] +
                 h * b[a] * PotentialEnergy(Q(q, a), Qdot(q, a)) * lag[i, a],
            1:length(c))
    end

    return DEL
end


"""
    DELSolve(DEL, q0, q01, q02, q1; ftol=1e-14)

Solve for the next triple `(q01_new, q02_new, q1_new)` given `(q0, q01, q02, q1)`.
"""
function DELSolve(DEL, q0, q01, q02, q1; ftol=1e-14)
    d = length(q0)
    guess = [q01; q02; q1]
    function objective(Q)
        Q1 = @view Q[1:d]
        Q2 = @view Q[d+1:2d]
        Q3 = @view Q[2d+1:end]
        eq1 = DEL(q0, q01, q02, q1, 4) + DEL(q1, Q1, Q2, Q3, 1) -
              rho/12 * (-6*q1 + (5/2)*(1+√5)*Q1 + (5/2)*(1-√5)*Q2 + Q3) -
              rho/12 * (-q0 + (5/2)*(-1+√5)*q01 - (5/2)*(1+√5)*q02 + 6*q1)
        eq2 = DEL(q1, Q1, Q2, Q3, 2) -
              5*rho/12 * ((1/2)*(-1-√5)*q1 + √5*Q2 + (1/2)*(1-√5)*Q3)
        eq3 = DEL(q1, Q1, Q2, Q3, 3) -
              5*rho/12 * ((1/2)*(-1+√5)*q1 - √5*Q1 + (1/2)*(1+√5)*Q3)
        return [eq1; eq2; eq3]
    end
    return nlsolve(objective, guess, autodiff=:forward, ftol=ftol)
end


"""
    DELSolve(DEL, q0, q01, q02, q1, steps; ftol=1e-14)

Compute a trajectory of `3*steps+1` configurations for the 4-stage method.
"""
function DELSolve(DEL, q0, q01, q02, q1, steps; ftol=1e-14)
    d = length(q0)
    trj = zeros(size(q0, 1), 3*steps + 1)
    trj[:, 1] = q0
    trj[:, 2] = q01
    trj[:, 3] = q02
    trj[:, 4] = q1
    @showprogress for j in 1:3:3*(steps-1)
        sol = DELSolve(DEL, @view(trj[:, j]), @view(trj[:, j+1]),
                       @view(trj[:, j+2]), @view(trj[:, j+3]), ftol=ftol).zero
        trj[:, j+4] = @view(sol[1:d])
        trj[:, j+5] = @view(sol[d+1:2d])
        trj[:, j+6] = @view(sol[2d+1:end])
    end
    return trj
end


function ConjugateMomenta_p0(DEL, q0, q01, q02, q1)
    return -DEL(q0, q01, q02, q1, 1) +
           rho/12 * (-6*q0 + (5/2)*(1+√5)*q01 + (5/2)*(1-√5)*q02 + q1)
end

function ConjugateMomenta_p1(DEL, q0, q01, q02, q1)
    return DEL(q0, q01, q02, q1, 4) -
           rho/12 * (-q0 + (5/2)*(-1+√5)*q01 - (5/2)*(1+√5)*q02 + 6*q1)
end


"""
    ConjugateMomenta(DEL, traj; steps)

Compute conjugate momenta along a trajectory matrix for the 4-stage method.
"""
function ConjugateMomenta(DEL, traj::Matrix; steps)
    p0 = [ConjugateMomenta_p0(DEL, @view(traj[:, 1]), @view(traj[:, 2]),
                               @view(traj[:, 3]), @view(traj[:, 4]))]
    ps = [ConjugateMomenta_p1(DEL, @view(traj[:, i+1]), @view(traj[:, i+2]),
                               @view(traj[:, i+3]), @view(traj[:, i+4]))
          for i in range(0, step=3, length=steps)]
    return stack([p0; ps], dims=2)
end


"""
    Step1(DEL, q0, p0; ftol=1e-14)

Solve for the first step `(q01, q02, q1)` given initial conditions `(q0, p0)`.
"""
function Step1(DEL, q0, p0; ftol=1e-14)
    d = length(q0)
    guess = [q0 + h*p0/3; q0 + h*p0/2; q0 + h*p0]
    function objective(Q)
        Q1 = @view Q[1:d]
        Q2 = @view Q[d+1:2d]
        Q3 = @view Q[2d+1:end]
        eq1 = p0 + DEL(q0, Q1, Q2, Q3, 1) -
              rho/12 * (-6*q0 + (5/2)*(1+√5)*Q1 + (5/2)*(1-√5)*Q2 + Q3)
        eq2 = DEL(q0, Q1, Q2, Q3, 2) -
              5*rho/12 * ((1/2)*(-1-√5)*q0 + √5*Q2 + (1/2)*(1-√5)*Q3)
        eq3 = DEL(q0, Q1, Q2, Q3, 3) -
              5*rho/12 * ((1/2)*(-1+√5)*q0 - √5*Q1 + (1/2)*(1+√5)*Q3)
        return [eq1; eq2; eq3]
    end
    return nlsolve(objective, guess, autodiff=:forward, ftol=ftol)
end


"""
    ComputeTrajectory(DEL, q0, p0, steps; ftol=1e-14)

Compute a full trajectory starting from initial conditions `(q0, p0)` for the 4-stage method.
"""
function ComputeTrajectory(DEL, q0, p0, steps; ftol=1e-14)
    sol = Step1(DEL, q0, p0).zero
    d = length(q0)
    q01 = sol[1:d]
    q02 = sol[d+1:2d]
    q1 = sol[2d+1:end]
    return DELSolve(DEL, q0, q01, q02, q1, steps, ftol=ftol)
end
