# linear-regresssion-with-one-feature
ML (linear regressin algorithm)

data=load('ex1data1.txt');
X=data(:,1);
y=data(:,2);
theta=[0;0];
m=length(X);
X=[ones(m,1),X];
x=X(:,2);
alpha=0.01;
iterations=1500;

computeCost(X,y,theta);
gradientDescent(X,y,theta,alpha,iterations);
plotData(x,y);
function plotData(x, y)
  
plot(x,y,'rx','MarkerSize',10);
axis([0 25 0 25])
ylabel('Profit in $10,000s');
xlabel('population of city in 10,000s')

end


function J = computeCost(X, y, theta)
  m=length(X);
  x=X(:,2);
  h=theta(1)+(theta(2)*x);
  J=(1/(2*m))*(sum(h-y).^2);
end

function [theta,J_history]=graidentDescent(X,y,theta,alpha,num_iters)
m = length(X);
J_history = zeros(num_iters, 1);
for iter = 1:num_iters,
  x=X(:,2);
  h=theta(1)+(theta(2)*x);
  theta_zero = theta(1) - alpha * (1/m) * sum(h-y);
  theta_one  = theta(2) - alpha * (1/m) * sum((h - y) .* x);
   theta = [theta_zero; theta_one];
   J_history(iter) = computeCost(X,y,theta);
end
  disp(min(J_history));
  theta
end

