function Coupled_Oscillator(omega)

function KineticEnergy(q,p)
[p[1]; p[2]]
end

function PotentialEnergy(q,p)
-[omega[1]*q[1] ; omega[2]*q[2] ]
end

return KineticEnergy, PotentialEnergy;
    
end