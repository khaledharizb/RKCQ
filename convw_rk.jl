using FFTW, LinearAlgebra

# --- Delta generating functions for Lobatto IIIC methods ---

function DeltaLobattoIIIC2(z)
    return [1    1-2*z
           -1    1]
end

function DeltaLobattoIIIC3(z)
    return [ 3   4  -1-6*z
            -1   0   1
             1  -4   3]
end

function DeltaLobattoIIIC4(z)
    a = √5
    return [ 6              (5/2)*(1+a)   -(5/2)*(-1+a)   1-12*z
            (1/2)*(-1-a)    0              a              (1/2)*(1-a)
            (1/2)*(-1+a)   -a              0              (1/2)*(1+a)
            -1             (5/2)*(-1+a)   -(5/2)*(1+a)    6]
end

# Lookup table: stages => (Butcher tableau A, b, c, Delta function)
const LOBATTO_IIIC = Dict(
    2 => (A = [1/2 -1/2; 1/2 1/2],
          b = [1/2, 1/2],
          c = [0.0, 1.0],
          delta = DeltaLobattoIIIC2),
    3 => (A = [1/6 -1/3 1/6; 1/6 5/12 -1/12; 1/6 2/3 1/6],
          b = [1/6, 2/3, 1/6],
          c = [0.0, 1/2, 1.0],
          delta = DeltaLobattoIIIC3),
    4 => (A = [1/12 -√5/12 √5/12 -1/12;
               1/12 1/4 (10-7*√5)/60 √5/60;
               1/12 (10+7*√5)/60 1/4 -√5/60;
               1/12 5/12 5/12 1/12],
          b = [1/12, 5/12, 5/12, 1/12],
          c = [0.0, (5-√5)/10, (5+√5)/10, 1.0],
          delta = DeltaLobattoIIIC4),
)

"""
    convw_rk(N, T, stages, Ks) -> (omega, c)

Compute RK convolution quadrature weights for Lobatto IIIC methods.

# Arguments
- `N::Int`: number of time steps
- `T::Real`: final time
- `stages::Int`: number of stages (2, 3, or 4)
- `Ks::Function`: transfer function K(s) evaluated at complex frequency

# Returns
- `omega`: real array of size (s, s, N+1) containing convolution weights
- `c`: abscissae of the RK method
"""
function convw_rk(N::Int, T::Real, stages::Int, Ks::Function)
    haskey(LOBATTO_IIIC, stages) || error("Unsupported stage count: $stages. Use 2, 3, or 4.")

    method = LOBATTO_IIIC[stages]
    s = stages
    dt = T / N
    Δ = method.delta

    Nfft = nextpow(2, 3 * N)
    λ = (1e-15)^(1 / (2 * Nfft))
    kv = 0:Nfft-1
    ξ = λ * cis.(2π * kv / Nfft)  # cis(θ) = exp(im*θ), avoids complex multiply

    # Compute generating function samples at all quadrature nodes
    omegalong = Array{ComplexF64}(undef, s, s, Nfft)
    for ll in 1:Nfft
        DV = Δ(ξ[ll])
        D, V = eigen(DV)
        omegalong[:, :, ll] = V * Diagonal(Ks.(D / dt)) * inv(V)
    end

    # FFT along the third axis and extract first N+1 coefficients
    omega = Array{Float64}(undef, s, s, N + 1)
    correction = λ .^ (-(0:N))  # undo the radius scaling

    for kk in 1:s, jj in 1:s
        coeffs = fft(@view omegalong[kk, jj, :])
        @views omega[kk, jj, :] = real.(correction .* coeffs[1:N+1]) / Nfft
    end

    return omega, method.c
end
