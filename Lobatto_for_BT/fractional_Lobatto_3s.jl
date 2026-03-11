function LobattoIIIC_3s(q₀ ,  p₀, α ,  N , T)

h = T/N;
    
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
sum(map(k -> q[k] * lag[k,i], 1:length(nodes)))
end

function Qdot(q,i)
sum(map(k -> q[k] * dlag[k,i], 1:length(nodes))) / h; 
end

f(t) = t^3  + 6 * t + (3.2 / gamma(0.5)) * t^2.5; # exact = t^3 for α=0.5    

τ = (0:N)*h;    
    
L(q,v) = (1/2)*(v^2)-(1/2)*(q^2); ∂ᵥL(q,v) = v;  ∂ₓL(t,q,v)  = - q + f(t); 

function  DL(i,q,k)      
  sum(map(a -> b[a] * ∂ᵥL(Q(q,a),Qdot(q,a))  * dlag[i,a] +
            h * b[a] * ∂ₓL(τ[k]+h*c[a],Q(q,a),Qdot(q,a)) * lag[i,a],1:length(c)))
end

function DEL0!(F , x , q₀ , p₀, k)  
    X = vcat(q₀,x[1],x[2]); 
    F[1] = p₀ + DL(1 , X, k) - b[1] * h^(1-α) * W[1,:,1]' * X ;  
 
    F[2] = DL(2 , X, k) - b[2] * h^(1-α) * W[2,:,1]' * X ;     
end 

init_gauss = [q₀ + h*p₀/2, q₀ + h*p₀]    
    
r = nlsolve((F,x) -> DEL0!(F , x , q₀ , p₀, 1) , init_gauss , autodiff = :forward, ftol=1e-14)

function DEL!(F , x , q , k)  
 X = vcat(q,x[1],x[2]);  
 qnew =  vcat(q[end],x[1],x[2]);  
Y = vcat(q[2],q[2],q);        
F[1] = DL(3,q[end-2:end],k-2) + DL(1,qnew,k-1) - 
     b[1]*h^(1-α) * sum(W[1,:,k-j]'*X[2*j-1:2*j+1] for j in 1:k-1) -
     b[3]*h^(1-α) * sum(W[3,:,k-j]'*Y[2*j-1:2*j+1] for j in 1:k-1) ;  
 
F[2] = DL(2,qnew, k-1) -  b[2]*h^(1-α) * sum(W[2,:,k-j]'*X[2*j-1:2*j+1] for j in 1:k-1); 
        
end 

q = [];
append!(q,q₀, r.zero); 

for k in 3:N+1
        
ini_gauss=[q[end-1],q[end]]   
        
r = nlsolve((F,x) -> DEL!(F , x , q, k) , ini_gauss , autodiff = :forward,  ftol=1e-16)
   

append!(q, r.zero)  

        
end 

return q;
    
end


