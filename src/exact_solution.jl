# Exact solutions for the 2-particle coupled damped oscillator system.
# Used for convergence validation and error measurement.

q1(t) = (4/155) * exp(-t/8) * (31 * cos(sqrt(31)/8 * t) + 5 * sqrt(31) * sin(sqrt(31)/8 * t))
q2(t) = (-1/62) * exp(-t/8) * (31 * cos(sqrt(31)/8 * t) + sqrt(31) * sin(sqrt(31)/8 * t))

p1(t) = (2/155) * exp(-t/8) * (31 * cos(sqrt(31)/8 * t) - 9 * sqrt(31) * sin(sqrt(31)/8 * t))
p2(t) = (2/sqrt(31)) * exp(-t/8) * sin(sqrt(31)/8 * t)
