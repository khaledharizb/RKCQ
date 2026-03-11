using FFTW, LinearAlgebra, BenchmarkTools

# ============================================================
# ORIGINAL implementation (as provided)
# ============================================================

function DeltaLabattoIIIC2_orig(z)
    return [1 1-2*z; -1 1]
end

function DeltaLabattoIIIC3_orig(z)
    return [3 4 -1-6*z; -1 0 1; 1 -4 3]
end

function DeltaLabattoIIIC4_orig(z)
    a = √5
    return [6 (5/2)*(1+a) -(5/2)*(-1+a) 1-12*z
        (1/2)*(-1-a) 0 a (1/2)*(1-a)
        (1/2)*(-1+a) -a 0 (1/2)*(1+a)
        -1 (5/2)*(-1+a) -(5/2)*(1+a) 6]
end

function convw_rk_original(N, T, RK, Ks::Function)
    dt = T / N

    if RK == 2
        s = 2
        A = [1/2 -1/2; 1/2 1/2]; b = [1/2, 1/2]; c = [0, 1]
        genfunc = "LabattoIIIC2"
    elseif RK == 3
        s = 3
        A = [1/6 -1/3 1/6; 1/6 5/12 -1/12; 1/6 2/3 1/6]
        b = [1/6, 2/3, 1/6]; c = [0, 1/2, 1]
        genfunc = "LabattoIIIC3"
    elseif RK == 4
        s = 4
        A = [1/12 -√5/12 √5/12 -1/12; 1/12 1/4 (10-7*√5)/60 √5/60;
             1/12 (10+7*√5)/60 1/4 -√5/60; 1/12 5/12 5/12 1/12]
        b = [1/12, 5/12, 5/12, 1/12]
        c = [0, (5-√5)/10, (5+√5)/10, 1]
        genfunc = "LabattoIIIC4"
    end

    Nmax = 3 * N
    Nmax = Int(2^ceil(log(Nmax) / log(2)))
    λ = (1e-15)^(1 / (2 * Nmax))
    kv = 0:Nmax-1
    xi = λ * exp.(1 * im * 2 * pi * kv / Nmax)

    omegalong = zeros(ComplexF64, (s, s, Nmax))
    omega = zeros(ComplexF64, (s, s, N + 1))
    omega0 = zeros(ComplexF64, s, s)

    for ll in 1:Nmax
        xx = xi[ll]
        DV = getfield(@__MODULE__, Symbol("Delta", genfunc, "_orig"))(xx)
        D, V = eigen(DV)
        temp = diagm(Ks.(D / dt))
        invV = inv(V)
        omegalong[:, :, ll] = V * temp * invV
    end

    for kk in 1:s
        for jj in 1:s
            omegaV = fft(omegalong[kk, jj, :])
            omegaV = λ .^ (-kv[1:N+1]) .* omegaV[1:N+1]
            omega0[kk, jj] = omegaV[1] / Nmax
            omega[kk, jj, :] = omegaV / Nmax
        end
    end
    return real(omega), c
end

# ============================================================
# IMPROVED implementation
# ============================================================

include("convw_rk.jl")

# ============================================================
# Benchmark
# ============================================================

# Simple transfer function for testing: K(s) = 1/s
Ks_test(s) = 1 / s

println("=" ^ 60)
println("Benchmarking convw_rk: Original vs Improved")
println("=" ^ 60)

for (stages, N_vals) in [(2, [64, 256, 1024]),
                          (3, [64, 256, 1024]),
                          (4, [64, 256, 1024])]
    println("\n--- $stages-stage Lobatto IIIC ---")
    for N in N_vals
        T = 1.0

        # Verify correctness: results should match
        ω_orig, c_orig = convw_rk_original(N, T, stages, Ks_test)
        ω_new, c_new = convw_rk(N, T, stages, Ks_test)
        maxerr = maximum(abs.(ω_orig - ω_new))

        println("\n  N = $N  (max difference = $maxerr)")

        print("    Original: ")
        b_orig = @benchmark convw_rk_original($N, $T, $stages, $Ks_test)
        show(stdout, MIME("text/plain"), b_orig)
        println()

        print("    Improved: ")
        b_new = @benchmark convw_rk($N, $T, $stages, $Ks_test)
        show(stdout, MIME("text/plain"), b_new)
        println()

        speedup = median(b_orig).time / median(b_new).time
        alloc_ratio = b_orig.allocs / b_new.allocs
        println("    Speedup: $(round(speedup, digits=2))x time, $(round(alloc_ratio, digits=2))x fewer allocations")
    end
end
