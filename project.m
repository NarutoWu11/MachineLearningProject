function project()
% perceptron_demo_wrapper - runs the perceptron model on a separable two 
% class dataset consisting of two dimensional data features. 
% The perceptron is run 3 times with 3 initial points to show the 
% recovery of different separating boundaries.  All points and recovered
% as well as boundaries are then visualized.

%%% load data %%%
[D,b] = load_data();
[P,k] = load_data1();
%[T,r] = load_data2();
%%% run perceptron for 3 initial points %%%

% Calculate fixed steplength - via Lipschitz constant (see Chap 9 for 
% explanation) - for use in all three runs
lam = 10^(0.3);        % regularization parameter 
L = 2*norm(diag(b)*D')^2;
alpha = 1/(L + 2*lam);        % step length
x0 = [1;2;3;4];    % initial point
x = grad_descent_soft_SVM(D,b,x0,alpha,lam);

lam1 = 10^(0.3);        % regularization parameter 
L1 = 2*norm(diag(k)*P')^2;
alpha1 = 1/(L1 + 2*lam1);        % step length
y0 = [1;2;3;4];    % initial point
y = grad_descent_soft_SVM1(P,k,y0,alpha1,lam1);

%%% plot everything, pts and lines %%%
plot_all(D',b,x,P',k,y);


%%% gradient descent function for perceptron %%%
function x = grad_descent_soft_SVM(D,b,x0,alpha,lam)
    % Initializations 
    x = x0;
    iter = 1;
    max_its = 3000;
    grad = 1;
    D = D';
    while  norm(grad) > 10^-6 && iter < max_its
        
        % form gradient and take step
        grad = 0;
        for i = 1 : size(D,1)
        grad = grad -2*b(i)*D(i,:)'*max(0, 1 - b(i)*D(i,:)*x);            % your code goes here!
        end
        grad = grad + 2*lam*[zeros(1,4);zeros(3,1), eye(3)]*x;
        x = x - alpha*grad;

        % update iteration count
        iter = iter + 1;
    end
end

function y = grad_descent_soft_SVM1(P,k,y0,alpha1,lam1)
    % Initializations 
    y = y0;
    iter = 1;
    max_its = 3000;
    grad = 1;
    P = P';
    while  norm(grad) > 10^-6 && iter < max_its
        
        % form gradient and take step
        grad = 0;
        for i = 1 : size(P,1)
        grad = grad -2*k(i)*P(i,:)'*max(0, 1 - k(i)*P(i,:)*y);            % your code goes here!
        end
        grad = grad + 2*lam1*[zeros(1,4);zeros(3,1), eye(3)]*y;
        y = y - alpha1*grad;

        % update iteration count
        iter = iter + 1;
    end
end

%%% plots everything %%%
function plot_all(A,b,x,C,d,y)
    
    % plot points 
    ind = find(b == 1);
    scatter3(A(ind,2),A(ind,3),A(ind,4),'Linewidth',2,'Markeredgecolor','b','markerFacecolor','none');
    hold on
    ind = find(b == -1 & d== -1);
    scatter3(A(ind,2),A(ind,3),A(ind,4),'Linewidth',2,'Markeredgecolor','r','markerFacecolor','none');
    scatter3(C(ind,2),C(ind,3),C(ind,4),'Linewidth',2,'Markeredgecolor','r','markerFacecolor','none');

    hold on
    ind = find(d == 1);
    scatter3(C(ind,2),C(ind,3),C(ind,4),'Linewidth',2,'Markeredgecolor','g','markerFacecolor','none');
    hold on
    
    range = 30;
    a1 = 100:2:range*10;
    a2 = 100:2:range*10;
    [A1, A2] = meshgrid(a1,a2);
     
    [m, n] = size(A1);
    for i = 1:m
        for j=1:n
            A = [A1(i,j);A2(i,j)];
            f1(i,j) =  (-x(1)-x(2)*A(1)-x(3)*A(2))/x(4);
        end
    end
   % for i = 1:m
   %     for j=1:n
    %        A = [C1(i,j);C2(i,j)];
   %         f1(i,j) =  (-x(1)-x(2)*A(1)-x(3)*A(2))/x(4);
    %    end
  %  end
    [m, n] = size(A1);
    for i = 1:m
        for j=1:n
            A = [A1(i,j);A2(i,j)];
            f2(i,j) =  (-y(1)-y(2)*A(1)-y(3)*A(2))/y(4);
        end
    end
    
    mesh(A1,A2,f1);
    %shading interp
    hold on;
    mesh(A1,A2,f2);
    
    %{
    % plot separators
    s =[min(A(:,2)):.01:max(A(:,2))];
    plot (s,(-x(1)-x(2)*s)/x(3),'k','linewidth',2);
    hold on

    plot (s,(-y(1)-y(2)*s)/y(3),'g','linewidth',2);
    hold on

    plot (s,(-z(1)-z(2)*s)/z(3),'m','linewidth',2);
    hold on

    set(gcf,'color','w');
    axis([ (min(A(:,2)) - 1) (max(A(:,2)) + 1) (min(A(:,3)) - 1) (max(A(:,3)) + 1)])
    box off
    
    % graph info labels
    xlabel('a_1','Fontsize',14)
    ylabel('a_2  ','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',0)
    %}

end

%%% loads data %%%
function [A,b] = load_data()
    data = xlsread('Seperator(1-2)new.xlsx');
    %data = load('nbaData.mat');
    %data = data.unnamed;
    A = data(:,1:4);
    A = A';
    b = data(:,5);
end

function [A,b] = load_data1()
    data = xlsread('Seperator(4-5)new.xlsx');
    %data = load('nbaData.mat');
    %data = data.unnamed;
    A = data(:,1:4);
    A = A';
    b = data(:,5);
end

end
