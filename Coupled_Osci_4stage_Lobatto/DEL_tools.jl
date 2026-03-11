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

function DEL(q0, q01, q02, q1,i);
   q = hcat(q0, q01, q02, q1);
   sum(map(a -> b[a] * KineticEnergy(Q(q,a),Qdot(q,a)) * dlag[i,a] + h * b[a] * PotentialEnergy(Q(q,a),Qdot(q,a)) * lag[i,a],1:length(c)))
end
    
return DEL
    
end


function DELSolve(DEL,q0, q01, q02 , q1;ftol=1e-14)
guess = [q01; q02; q1]; d = length(q0);
objective(Q) = [DEL(q0, q01, q02, q1,4)+DEL(q1,Q[1:d],Q[d+1:2d],Q[2d+1:end],1)-rho/12*(-6*q1+(5/2)*(1+√5)*Q[1:d]+(5/2)*(1 - √5)*Q[d+1:2d]+Q[2d+1:end])-rho/12 *(-q0 + (5/2)*(-1+√5)*q01-(5/2)*(1+√5)*q02 +  6*q1); DEL(q1,Q[1:d],Q[d+1:2d],Q[2d+1:end],2) - 5*rho/12 *  ((1/2)*(-1
    -√5)*q1+√5*Q[d+1:2d] + (1/2)*(1 - √5)*Q[2d+1:end]); DEL(q1,Q[1:d],Q[d+1:2d],Q[2d+1:end],3) -
        5*rho/12 *  ((1/2)*(-1+√5)*q1-√5*Q[1:d] + (1/2)*(1 + √5)*Q[2d+1:end]) ]
    return nlsolve(objective, guess,autodiff = :forward,ftol=ftol)
end


function DELSolve(DEL,q0,q01,q02,q1,steps;ftol=1e-14)
       d = length(q0);
    trj = zeros((size(q0)[1],3*steps+1))
    trj[:,1]=q0
    trj[:,2]=q01
    trj[:,3]=q02
    trj[:,4]=q1
 @showprogress   for j = 1:3:3*(steps-1)     
       sol = DELSolve(DEL,trj[:,j],trj[:,j+1],trj[:,j+2],trj[:,j+3],ftol=ftol).zero
    trj[:,j+4] = sol[1:d]
    trj[:,j+5] = sol[d+1:2d]
    trj[:,j+6] = sol[2d+1:end]
    end
    
    return trj
end

function ConjugateMomenta_p0(DEL,q0,q01,q02,q1)
    return  -DEL(q0,q01,q02,q1,1) + rho/12*(-6*q0+(5/2)*(1+√5)*q01+(5/2)*(1 - √5)*q02+q1) ;
end

function ConjugateMomenta_p1(DEL,q0,q01,q02,q1)
    return   DEL(q0,q01,q02,q1,4) -rho/12 *(-q0 + (5/2)*(-1+√5)*q01-(5/2)*(1+√5)*q02 +  6*q1);
end



function ConjugateMomenta(DEL,traj::Matrix;steps)
   p0 = [ConjugateMomenta_p0(DEL,traj[:,1],traj[:,2],traj[:,3],traj[:,4])];
   ps = [ConjugateMomenta_p1(DEL,traj[:,i+1],traj[:,i+2],traj[:,i+3],traj[:,i+4]) for i in range(0, step = 3, length=steps)];
   all_ps = [p0; ps];
    return stack(all_ps,dims=2)
end



function Step1(DEL,q0,p0;ftol=1e-14)
d = length(q0); guess = [q0 + h * p0 / 3 ; q0 + h * p0 / 2 ; q0 + h * p0];
objective(Q) = [p0 + DEL(q0,Q[1:d],Q[d+1:2d],Q[2d+1:end],1)-rho/12 *(-6*q0+(5/2)*(1+√5)*Q[1:d]+(5/2)*(1 
    - √5)*Q[d+1:2d]+Q[2d+1:end]) ; DEL(q0,Q[1:d],Q[d+1:2d],Q[2d+1:end],2)- 5*rho/12 *  ((1/2)*(-1
                -√5)*q0+√5*Q[d+1:2d] + (1/2)*(1-√5)*Q[2d+1:end]) ; DEL(q0,Q[1:d],Q[d+1:2d],Q[2d+1:end],3)- 
         5*rho/12 *  ((1/2)*(-1+√5)*q0-√5*Q[1:d] + (1/2)*(1 + √5)*Q[2d+1:end]) ];   
    return nlsolve(objective,guess,autodiff =:forward,ftol=ftol)
end

function ComputeTrajectory(DEL,q0,p0,steps;ftol=1e-14)
   sol = Step1(DEL,q0,p0).zero; d = length(q0);
   q01 = sol[1:d] ; q02 = sol[d+1:2d]; q1 = sol[2d+1:end];
    return DELSolve(DEL,q0,q01,q02,q1,steps,ftol=ftol)
end