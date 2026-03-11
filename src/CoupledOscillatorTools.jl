"""
    Coupled_Oscillator(omega)

Construct kinetic and potential energy functions for a coupled oscillator system
with natural frequencies `omega`.

Returns `(KineticEnergy, PotentialEnergy)` function pair.
"""
function Coupled_Oscillator(omega)

    function KineticEnergy(q, p)
        return [p[1]; p[2]]
    end

    function PotentialEnergy(q, p)
        return -[omega[1] * q[1]; omega[2] * q[2]]
    end

    return KineticEnergy, PotentialEnergy
end
