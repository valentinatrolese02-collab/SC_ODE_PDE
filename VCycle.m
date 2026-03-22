% exact solution and RHS
u=@(x,y) exp(pi*x).*sin(pi*y)+0.5*(x.*y).^2;
f=@(x,y) x.^2+y.^2;
m=2^6-1;
U =zeros(m*m,1);
F =form_rhs(m,f,u); %% TODO: Form the right-hand side
epsilon = 1.0E-10;
for i=1:100
    R =F+Amult(U,m);
    fprintf('*** Outer iteration: %3d, rel. resid.: %e\n', ...
        i, norm(R,2)/norm(F,2));
    if(norm(R,2)/norm(F,2) < epsilon)
        break;
    end
    U=Vcycle(U,omega,3,m,F);
    plotU(m,U);
    pause(.5);
end

function Unew=Vcycle(U,omega,nsmooth,m,F)
% Approximately solve: A*U = F
h=1.0/(m+1);
l2m=log2(m+1);
assert(l2m==round(l2m));
assert(length(U)==m*m);
if(m==1)
    % if we are at the coarsest level
    % TODO: solve the only remaining equation directly!
else
    % 1. TODO: pre-smooth the error
    %    perform <nsmooth> Jacobi iterations
    % 2. TODO: calculate the residual
    % 3. TODO: coarsen the residual
    % 4. recurse to Vcycle on a coarser grid
    mc=(m-1)/2;
    Ecoarse=Vcycle(zeros(mc*mc,1),omega,nsmooth,mc,-Rcoarse);
    % 5. TODO: interpolate the error
    % 6. TODO: update the solution given the interpolated error
    % 7. TODO: post-smooth the error
    %    perform <nsmooth> Jacobi iterations
end
end

function plotU(m,U)
h=1/(m+1);
x=linspace(1/h,1-1/h,m);
y=linspace(1/h,1-1/h,m);
[X,Y]=meshgrid(x,y);
surf(X, Y, reshape(U,[m,m])');
shading interp;
title('Computed solution');
xlabel('x');
ylabel('y');
zlabel('U');
end