function DiscreteEL(KineticEnergy,PotentialEnergy,nodes,b,c,h)

function Basis(j,t)
        y=1;
         for i in 1:length(nodes)
            if i!=j
             y = y*(t-nodes[i])/(nodes[j]-nodes[i]);    
            end
         end
     return y;
  end 

function dBasis(j,t)
   ForwardDiff.derivative(t->Basis(j,t),t)
end  

lag = map(j-> Basis.(j,c), 1:length(nodes)); lag = stack(lag,dims=1);

dlag = map(j-> dBasis.(j,c), 1:length(nodes)); dlag = stack(dlag,dims=1);

function Q(q,i)
sum(map(k -> q[:,k] * lag[k,i], 1:length(nodes)))
end

function Qdot(q,i)
sum(map(k -> q[:,k] * dlag[k,i], 1:length(nodes))) / h; 
end

function DEL(q0, q1,i);
   q = hcat(q0, q1);
   sum(map(a -> b[a] * KineticEnergy(Q(q,a),Qdot(q,a)) * dlag[i,a] + h * b[a] * PotentialEnergy(Q(q,a),Qdot(q,a)) * lag[i,a],1:length(c)))
end
    
return DEL
    
end


function DELSolve(DEL,q0, q1;ftol=1e-14)
    guess = q0; d = length(q0);
objective(Q) = DEL(q0, q1,2)+DEL(q1,Q[1:d],1) -rho/2 *  (Q[1:d]-q1) - rho/2 * (q1-q0)   
    return nlsolve(objective, guess,autodiff = :forward, method = :newton,ftol=ftol)
end


function DELSolve(DEL,q0,q1,steps;ftol=1e-14)
       d = length(q0);
    trj = zeros((size(q0)[1],steps+1))
    trj[:,1]=q0
    trj[:,2]=q1
 @showprogress   for j = 1:(steps-1)     
       sol = DELSolve(DEL,trj[:,j],trj[:,j+1],ftol=ftol).zero
    trj[:,j+2] = sol[1:d]
    end
    
    return trj
end

function ConjugateMomenta_p0(DEL,q0,q1)
    return  -DEL(q0,q1,1) + rho/2 *  (q1-q0)   ;
end

function ConjugateMomenta_p1(DEL,q0,q1)
    return   DEL(q0,q1,2) - rho/2 *  (q1-q0);
end



function ConjugateMomenta(DEL,traj::Matrix;steps)
   p0 = [ConjugateMomenta_p0(DEL,traj[:,1],traj[:,2])];
   ps = [ConjugateMomenta_p1(DEL,traj[:,i],traj[:,i+1]) for i in range(1, length=steps)];
   all_ps = [p0; ps];
    return stack(all_ps,dims=2)
end



function Step1(DEL,q0,p0;ftol=1e-14)
    d = length(q0); guess = q0 + h * p0;
objective(Q) = p0 + DEL(q0,Q[1:d],1) - rho/2 *  (Q[1:d]-q0)   
    return nlsolve(objective,guess,autodiff =:forward,method = :newton,ftol=ftol)
end

function ComputeTrajectory(DEL,q0,p0,steps;ftol=1e-14)
   sol = Step1(DEL,q0,p0).zero; d = length(q0);
   q1 = sol[1:d] ;
    return DELSolve(DEL,q0,q1,steps,ftol=ftol)
end