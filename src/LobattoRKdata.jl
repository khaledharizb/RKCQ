"""
    LobattoRKdata(RK)

Return the Butcher tableau `(A, b, c)` for Lobatto IIIC Runge-Kutta methods.

- `RK == 1`: 2-stage, order 2, stage order 1
- `RK == 2`: 3-stage, order 4, stage order 2
- `RK == 3`: 4-stage, order 6, stage order 3
"""
function LobattoRKdata(RK)
    if RK == 1  # Lobatto IIIC order 2, stage order 1
        A = [1/2  -1/2
             1/2   1/2]
        b = [1/2, 1/2]
        c = [0.0, 1.0]

    elseif RK == 2  # Lobatto IIIC order 4, stage order 2
        A = [1/6  -1/3   1/6
             1/6   5/12 -1/12
             1/6   2/3   1/6]
        b = [1/6, 2/3, 1/6]
        c = [0.0, 1/2, 1.0]

    elseif RK == 3  # Lobatto IIIC order 6, stage order 3
        A = [1/12  -√5/12          √5/12         -1/12
             1/12   1/4            (10-7√5)/60     √5/60
             1/12  (10+7√5)/60     1/4            -√5/60
             1/12   5/12           5/12            1/12]
        b = [1/12, 5/12, 5/12, 1/12]
        c = [0.0, (5-√5)/10, (5+√5)/10, 1.0]

    else
        error("Unsupported RK method: $RK. Use 1, 2, or 3.")
    end

    return A, b, c
end
