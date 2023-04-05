function [loss,G]=loss(w)
B = [0 113 188 ]/norm([0 113 188 ]);
global model counter

for i=1:length(model.layers)-1
    m = model.layer(i).iw;
    n = model.layer(i).ws;
    
    model.layer(i).W = reshape(w(m(1):m(2)),n(1),n(2));
    
    m = model.layer(i).ib;
    n = model.layer(i).bs;
    
    model.layer(i).b = reshape(w(m(1):m(2)),n(1),n(2));
end

loss = 0;
if model.flagd > 0
    y_pred = forward_pass(model.x);
    lossd = sum((y_pred - model.y).^2)/(2*model.Nd);
    loss = loss + model.flagd * lossd;
end

if model.flagm > 0
    L = D2y(model.xm); % Laplacian
    pde_res = model.k * L + q(model.xm')';
    lossm = sum(pde_res.^2, "all")/(2*model.Nm);
    loss = loss + model.flagm * lossm;
end

G = Gradloss;

counter = counter + 1;
if (mod(counter,100)==0)
    N = model.xstar_size;
    ystar = forward_pass(model.xstar);
    ystar = reshape(ystar,N(2),N(1));
    X1 = reshape(model.xstar(1,:),N(2),N(1));
    X2 = reshape(model.xstar(2,:),N(2),N(1));
    %  surf(X1,X2,model.ystar_t,'FaceColor',[.5 .5 .5]);hold on
    %  plot3(model.x(1,:),model.x(2,:),model.y,'o','MarkerSize',15);
    %  surf(X1,X2,ystar);hold off
    subplot(1,2,1)
    contourf(X1,X2,model.ystar_t);hold on
    plot(model.x(1,:),model.x(2,:),'o','MarkerSize',15);
    colorbar
    axis equal
    xlabel('$x_1$')
    ylabel('$x_2$')
    title('Truth: Finite-Difference')
    set(gca,'FontSize',20)
    hold off
    % ---------------------------------------------------------------------
    subplot(1,2,2)
    contourf(X1,X2,ystar);hold on
    plot(model.x(1,:),model.x(2,:),'o','MarkerSize',15);
    axis equal
    colorbar
    if (model.flagm && model.flagd)
        title('Neural Network: Data+Physics')
    end
    if (~model.flagm && model.flagd)
        title('Neural Network: Data')
    end
    if (model.flagm && ~model.flagd)
        title('Neural Network: Physics')
    end
    hold off
    xlabel('$x_1$')
    ylabel('$x_2$')
    set(gca,'FontSize',20)
    drawnow
end
