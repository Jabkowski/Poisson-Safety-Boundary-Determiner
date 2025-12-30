function [h, dhdx, dhdy, grid] = GeneratePoissonSafetyFunction(boundry, objects_list)
    %% Define boundries
    gd = boundry; % external boundry
    sf = "R1";
    ns = 'R1';
    % Add circle by circle
    for index = 1:length(objects_list)
        object = objects_list{index};
        ci = [1 object.c(1) object.c(2) object.r 0 0 0 0 0 0]';
        gd = [gd ci];
        ns = char(ns, ['C' num2str(index)]);
        sf = sf + " - C" + string(index);
    end
    ns = ns';

    % Define PDE solver
    N = 2;
    model = createpde(N);

    % Load geometry
    [g, ~] = decsg(gd, sf, ns);
    geometryFromEdges(model, g);

    % Define equation
    specifyCoefficients(model, 'm', 0, 'd', 0, 'c', 1, 'a', 0, 'f', [0;0]);

    % Boundry condition on objects
    function bcval = bc_location(location, ~)
        % location.x, location.y - border points
        x = location.x;
        y = location.y;
        np = numel(x);
        bcval = zeros(2, np);   % all points: [u_x; u_y]
        
        for k = 1:np
            for object_index = 1:length(objects_list)
                dx = x(k) - objects_list{object_index}.c(1);
                dy = y(k) - objects_list{object_index}.c(2);
                d = sqrt(dx^2 + dy^2);
                if d <= objects_list{object_index}.r + 1e-6
                    bcval(:,k) = 2000 * [x(k) - objects_list{object_index}.c(1); y(k) - objects_list{object_index}.c(2)];
                end
            end

            bcval(:,k) = [0.01; 0.01];
        end
    end

    % Set boundry conditions
    applyBoundaryCondition(model, ...
        'dirichlet', ...
        'Edge', 1:model.Geometry.NumEdges, ...
        'u', @bc_location);

    % --- Generate mesh
    generateMesh(model, 'Hmax', 0.01);

    % --- Solve PDE
    result = solvepde(model);


    %% Solve Poisson equation for h, dhdx and dhdy
    ux = result.NodalSolution(:,1);
    uy = result.NodalSolution(:,2);
    f_nodes = sqrt(ux.^2 + uy.^2);

    Xnodes = model.Mesh.Nodes(1,:)';
    Ynodes = model.Mesh.Nodes(2,:)';

    Finterp = scatteredInterpolant(Xnodes, Ynodes, f_nodes, 'linear', 'nearest');

    % Define equation
    model_h = createpde(1);
    p = model.Mesh.Nodes;        % 2 x Nnodes
    t = model.Mesh.Elements;     % 3 x Nelements
    geometryFromMesh(model_h, p, t);

    specifyCoefficients(model_h, ...
        'm', 0, ...
        'd', 0, ...
        'c', -1, ...
        'a', 0, ...
        'f', @(location,state) f_for_toolbox(location, Finterp));

    function fq = f_for_toolbox(location, Finterp)
        xq = location.x; yq = location.y;
        fq_phys = Finterp(xq, yq);
        fq = -fq_phys(:)';
    end

    applyBoundaryCondition(model_h, ...
        'dirichlet', ...
        'Edge', 1:model_h.Geometry.NumEdges, ...
        'u', 0);

    result_h = solvepde(model_h);

    %% Convert to grids
    h = result_h.NodalSolution; % solution in FEM modes
    [p, ~, ~] = meshToPet(model_h.Mesh);
    
    % Extract nodes
    x = p(1, :)';
    y = p(2, :)';
    
    % Set grid
    n = 512;
    nx = n;
    ny = n;
    xq = linspace(min(x), max(x), nx);
    yq = linspace(min(y), max(y), ny);
    [Xq, Yq] = meshgrid(xq, yq);
    
    % interpolate h(x,y)
    h = griddata(x, y, h, Xq, Yq, 'linear');
    
    [dhdx, dhdy] = gradient(h, xq, yq);

    %% Remove NaN
    h(isnan(h)) = 0;
    dhdx(isnan(dhdx)) = 0;
    dhdy(isnan(dhdy)) = 0;
    grid = h > 0;
end



