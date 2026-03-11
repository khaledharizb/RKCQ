using FFTW, LinearAlgebra, Printf

function  DeltaLabattoIIIC2(z) # 2-stage Radau
    return [1 1-2*z
         -1 1]; 
end    

function  DeltaLabattoIIIC3(z) # 3-stage Radau
    return [3 4  -1-6*z
        -1  0 1
        1  -4 3]; 
end    

function  DeltaLabattoIIIC4(z) # 4-stage Radau
    a = √5;
return [6 (5/2)*(1+a)  -(5/2)*(-1 + a)  1-12*z
    (1/2)*(-1-a) 0 a (1/2)*(1 - a)
    (1/2)*(-1 + a) -a 0  (1/2)*(1 + a)
    -1 (5/2)*(-1+a) -(5/2)*(1+a)   6]
end

# compute RK convolution weights

function convw_rk(N,T,RK,Ks::Function) 

dt = T/N;

if  (RK==2)   

s=2;
        
A = [1/2 -1/2 ; 1/2 1/2] ; b = [1/2 , 1/2]; c = [0 , 1];
genfunc = "LabattoIIIC2"  

elseif  (RK==3)
    s=3;
    A = [1/6  -1/3 1/6; 1/6 5/12 -1/12; 1/6 2/3 1/6]; b = [1/6, 2/3, 1/6];
        c = [0, 1/2 , 1]; 
    genfunc = "LabattoIIIC3"

  elseif  RK==4  # Labatto IIIC order 6, stage order 4    
      s=4;      
    A = [1/12 -√5/12 √5/12 -1/12 ; 1/12 1/4 (10-7*√5)/60 √5/60 ; 1/12 (10+7*√5)/60 1/4  -√5/60 ; 1/12 5/12 5/12 1/12];
    b = [1/12, 5/12, 5/12, 1/12];
    c = [0 , (5-√5)/10 , (5+√5)/10 , 1];
   genfunc = "LabattoIIIC4"       
    
end

Nmax = 3*N;                       # Nmax Power of 2 bigger than 2*N;
Nmax = 2^ceil(log(Nmax)/log(2));

λ = (1e-15)^(1/(2*Nmax));               		 
kv = 0:Nmax-1;                           
xi =  λ * exp.(1*im*2*pi*kv/Nmax);

omegalong=zeros(ComplexF64,(s,s,Int(Nmax)))

omega  = zeros(ComplexF64,(s,s,N+1))
omega0 = zeros(ComplexF64,s,s)


for ll=1:Int(Nmax)
xx = xi[ll];
DV  =  getfield(@__MODULE__, Symbol("Delta", genfunc))(xx)

D, V = eigen(DV)
temp = diagm(Ks.(D/dt));
invV = inv(V);
omegalong[:,:,ll] = V*temp*invV
end

for kk =1:s
    for jj = 1:s
omegaV = fft(omegalong[kk,jj,:])
omegaV = λ .^(-kv[1:N+1]) .*omegaV[1:N+1]
omega0[kk,jj] = omegaV[1]/Nmax;
omega[kk,jj,:] = omegaV/Nmax;
    end
end
    return real(omega), c # return to RK convolution matrix weights W[:, :, 1], W[:, :, 2] ...
end
