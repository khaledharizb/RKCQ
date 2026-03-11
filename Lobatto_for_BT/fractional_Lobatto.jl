function LobattoIIIC(q₀, p₀, α, N, T, f)

    h = T / N
    s = length(nodes)
    m = s - 1

    DL = DiscreteEL(f, nodes, b, c, h, N)

    return ComputeTrajectory(DL, q₀, p₀, N)

end
