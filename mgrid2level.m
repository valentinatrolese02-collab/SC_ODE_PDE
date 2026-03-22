a = .5;
psi     = @(x) 20*pi*x.^3;
psidot  = @(x) 3*20*pi*x.^2;
psiddot = @(x) 2*3*20*pi*x;

f = @(x) -20 + a*psiddot(x).*cos(psi(x)) - a*psidot(x).^2.*sin(psi(x));
u = @(x) 1 + 12*x - 10*x.^2 + a*sin(psi(x));

m=255;
h=1/(m+1);

A=spdiags(repmat([1, -2, 1]/h^2,m,1),-1:1,m,m);
X   =linspace(0+h,1-h,m)';
F   =f(X);
F(1)=F(1)-u(0)/h^2; F(m)=F(m)-u(1)/h^2;
Uhat=u(X);
Ehat=A\F-Uhat;

M=diag(diag(A));
N=M-A;
G=M\N;
b=M\F;

omega = 2/3;

U2=1+2*X;
for i=1:10,
  U2 = (1-omega)*U2+omega*((G*U2)+b);
  E2 = U2-Uhat;
  subplot(1,2,1),
  plot(X,Uhat,'b-', ...
       X,U2,   'gx');
  set(gca,'fontsize',16);
  xlabel('x');
  ylabel('U');
  title(sprintf('Iter=%4d', i));
  subplot(1,2,2),
  plot(X,Ehat,'b-',...
       X,E2,'gx');
  set(gca,'fontsize',16);
  xlabel('x');
  ylabel('E');
  title(sprintf('Iter=%4d', i));
  set(gcf,'color',[1,1,1]);
  pause(1);
end
pause;
% calculate residual
r = F-A*U2;
% coarsen
m_coarse = (m-1)/2;
h_coarse = 1/(m_coarse+1);
r_coarse = r(2:2:end);
assert(length(r_coarse)==m_coarse);
A_coarse=spdiags(repmat([1, -2, 1]/h_coarse^2,m_coarse,1),-1:1,...
                 m_coarse,m_coarse);
% solve the coarse problem
% normally we would do multigrid again
% here I just use the direct solver
e_coarse = A_coarse\(-r_coarse);
% project back on the fine grid
e=zeros(size(r));
e(2:2:end)=e_coarse;
for i=1:2:m,
  if(i>1)
    e_left = e(i-1);
  else
    e_left = 0;
  end
  if(i<m)
    e_right = e(i+1);
  else
    e_right = 0;
  end
  e(i) = (e_left+e_right)/2;
end
U2=U2-e;
E2=U2-Uhat;
subplot(1,2,1),
plot(X,Uhat,'b-', ...
     X,U2,   'gx');
set(gca,'fontsize',16);
xlabel('x');
ylabel('U');
title('After coarse grid projection');
subplot(1,2,2),
plot(X,Ehat,'b-',...
     X,E2,'gx');
set(gca,'fontsize',16);
xlabel('x');
ylabel('E');
title('After coarse grid projection');
set(gcf,'color',[1,1,1]);
pause;
% smooth the error again
for i=1:10,
  U2 = (1-omega)*U2+omega*((G*U2)+b);
  E2 = U2-Uhat;
  subplot(1,2,1),
  plot(X,Uhat,'b-', ...
       X,U2,   'gx');
  set(gca,'fontsize',16);
  xlabel('x');
  ylabel('U');
  title(sprintf('Iter=%4d', i));
  subplot(1,2,2),
  plot(X,Ehat,'b-',...
       X,E2,'gx');
  set(gca,'fontsize',16);
  xlabel('x');
  ylabel('E');
  title(sprintf('Iter=%4d', i));
  set(gcf,'color',[1,1,1]);
  pause(1);
end
