function DiscreteEL(f, nodes, b, c, h, N)

    s = length(nodes)

    function Basis(j, t)
        y = 1
        for i in 1:s
            if i != j
                y = y * (t - nodes[i]) / (nodes[j] - nodes[i])
            end
        end
        return y
    end

    function dBasis(j, t)
        ForwardDiff.derivative(t -> Basis(j, t), t)
    end

    lag  = stack(map(j -> Basis.(j, c),  1:s), dims=1)
    dlag = stack(map(j -> dBasis.(j, c), 1:s), dims=1)

    function Q(q, i)
        sum(map(k -> q[k] * lag[k, i], 1:s))
    end

    function Qdot(q, i)
        sum(map(k -> q[k] * dlag[k, i], 1:s)) / h
    end

    τ = (0:N) * h

    ∂ᵥL(q, v) = v
    ∂ₓL(t, q, v) = -q + f(t)

    function DL(q, i, k)
        sum(map(a -> b[a] * ∂ᵥL(Q(q, a), Qdot(q, a)) * dlag[i, a] +
                     h * b[a] * ∂ₓL(τ[k] + h * c[a], Q(q, a), Qdot(q, a)) * lag[i, a], 1:s))
    end

    return DL

end


function Step1(DL, q₀, p₀; ftol=1e-14)
    function objective(x)
        X = vcat(q₀, x...)
        F = similar(x)
        F[1] = p₀ + DL(X, 1, 1) - b[1] * h^(1 - α) * W[1, :, 1]' * X
        for i in 2:m
            F[i] = DL(X, i, 1) - b[i] * h^(1 - α) * W[i, :, 1]' * X
        end
        return F
    end
    guess = [q₀ + i * h * p₀ / m for i in 1:m]
    return nlsolve(objective, guess, autodiff=:forward, ftol=ftol)
end


function DELSolve(DL, q, k; ftol=1e-16)
    function objective(x)
        X    = vcat(q, x...)
        qnew = vcat(q[end], x...)
        Y    = vcat(q[1:m], q)
        F = similar(x)

        F[1] = DL(q[end-m:end], s, k - 2) + DL(qnew, 1, k - 1) -
               b[1] * h^(1 - α) * sum(W[1, :, k-j]' * X[m*(j-1)+1:m*j+1] for j in 1:k-1) -
               b[s] * h^(1 - α) * sum(W[s, :, k-j]' * Y[m*(j-1)+1:m*j+1] for j in 1:k-1)

        for i in 2:m
            F[i] = DL(qnew, i, k - 1) -
                   b[i] * h^(1 - α) * sum(W[i, :, k-j]' * X[m*(j-1)+1:m*j+1] for j in 1:k-1)
        end
        return F
    end
    guess = q[end-m+1:end]
    return nlsolve(objective, guess, autodiff=:forward, ftol=ftol)
end


function DELSolve(DL, q₀, q_init, N; ftol=1e-14)
    q = []
    append!(q, q₀, q_init)
    for k in 3:N+1
        sol = DELSolve(DL, q, k, ftol=ftol)
        append!(q, sol.zero)
    end
    return q
end


function ComputeTrajectory(DL, q₀, p₀, N; ftol=1e-14)
    sol = Step1(DL, q₀, p₀, ftol=ftol)
    return DELSolve(DL, q₀, sol.zero, N, ftol=ftol)
end
