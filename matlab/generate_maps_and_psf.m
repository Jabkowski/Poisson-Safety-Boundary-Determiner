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
        % Define external boundry [0, 1] x [0, 1]
        boundry = [3 4 0 1 1 0 0 0 1 1]';
        
        % Define objects
        rng(map_index); % randomnumber generator seed
        max_objects_number = 50;
        objects_number = ceil(max_objects_number * rand());
        
        % Zmieniamy na dynamiczną listę, bo niektóre próby mogą zostać odrzucone
        objects_list = {};
            
        if mod(map_index, 20) ~= 0
            for index = 1:objects_number
                
                valid_object = false;
                attempts = 0;
                max_attempts = 200; % Zabezpieczenie przed nieskończoną pętlą, gdy brakuje miejsca
                
                while ~valid_object && attempts < max_attempts
                    attempts = attempts + 1;
                    
                    % 1. Losowanie potencjalnego obiektu
                    potential_c = rand(2, 1);
                    potential_r = 0.1 * rand() + 0.02;
                    
                    % 2. Sprawdzenie warunku brzegowego (czy mieści się w kwadracie [0,1]x[0,1])
                    % Dodajemy minimalny margines bezpieczeństwa (np. 1e-4) od krawędzi zewnętrznej
                    margin = 1e-4;
                    if (potential_c(1) - potential_r < margin) || (potential_c(1) + potential_r > 1 - margin) || ...
                       (potential_c(2) - potential_r < margin) || (potential_c(2) + potential_r > 1 - margin)
                        continue; % Nie pasuje, losuj od nowa
                    end
                    
                    % 3. Sprawdzenie nakładania się / inkluzji z dotychczasowymi obiektami
                    overlap = false;
                    for j = 1:length(objects_list)
                        existing_obj = objects_list{j};
                        % Odległość między środkami okręgów
                        dist_centers = norm(potential_c - existing_obj.c);
                        
                        % Okręgi nakładają się lub jeden jest w drugim, jeśli:
                        % odległość środków < suma ich promieni (+ mały margines separacji)
                        if dist_centers < (potential_r + existing_obj.r + 0.005)
                            overlap = true;
                            break; % Wykryto kolizję, przerwij sprawdzanie reszty
                        end
                    end
                    
                    % Jeśli przeszedł oba testy, obiekt jest poprawny
                    if ~overlap
                        valid_object = true;
                        objects_list{end+1} = struct('c', potential_c, 'r', potential_r); %#ok<AGROW>
                    end
                end
                
                % Jeśli skończyło się miejsce na mapie, nie ma sensu losować kolejnych kół
                if attempts >= max_attempts
                    warning('Mapa %d: Osiągnięto limit prób upchnięcia kół. Wygenerowano %d z %d planowanych.', ...
                        map_index, length(objects_list), objects_number);
                    break;
                end
            end
        else
            objects_list = {};
        end
        
        % Transponujemy listę do formatu kolumnowego, jak w oryginalnym kodzie (opcjonalnie)
        objects_list = objects_list';
        
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
            sprintf('Progress: %d %%, index = %d', floor(map_index / generated_maps_number * 100), map_index));
    end
    close(wb); % Pamiętajmy o zamknięciu waitbara na koniec
end