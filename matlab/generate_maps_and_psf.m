function generate_maps_and_psf(varargin)

    % Validate input arguments
    ip = inputParser;
    addOptional(ip, 'file_name', "training_data_512x512.h5", @isstring);
    addOptional(ip, 'generated_maps_number', 10, @isnumeric);
    parse(ip, varargin{:});   
    
    file_name = ip.Results.file_name;
    generated_maps_number = ip.Results.generated_maps_number;

    addpath source
    
    wb = waitbar(0, 'Starting');

    for map_index = 1:generated_maps_number
        % Define external boundry
        boundry = [3 4 0 1 1 0 0 0 1 1]';
        
        % Define objects
        rng(map_index); % randomnumber generator seed
        max_objects_number = 50;
        objects_number = ceil(max_objects_number * rand());
        objects_list = cell(objects_number, 1);
            
        for index = 1:objects_number
            objects_list{index} = struct('c', rand(2, 1), 'r', 0.02);
        end
        
        % Generate h, dhx and dhy
        [h, dhdx, dhdy, grid] = GeneratePoissonSafetyFunction(boundry, objects_list);
    
        % Write data
        index_string = sprintf('%06d', map_index);
        
        h5create(file_name, "/grid/" + index_string, size(grid), 'Datatype', 'uint8');
        h5write(file_name, "/grid/" + index_string, uint8(grid));

        h5create(file_name, "/h/" + index_string, size(h), 'Datatype', 'single');
        h5write(file_name, "/h/" + index_string, single(h));
        
        h5create(file_name, "/dhdx/" + index_string, size(dhdx), 'Datatype', 'single');
        h5write(file_name, "/dhdx/" + index_string, single(dhdx));
    
        h5create(file_name, "/dhdy/" + index_string, size(dhdy), 'Datatype', 'single');
        h5write(file_name, "/dhdy/" + index_string, single(dhdy));

        waitbar(map_index / generated_maps_number, ...
            wb, ...
            sprintf('Progress: %d %%', floor(map_index / generated_maps_number * 100)));
    end

end