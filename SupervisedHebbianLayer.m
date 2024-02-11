classdef SupervisedHebbianLayer
    properties
        weights
        numInputs
        outputSize
        transferFunc
        alpha
    end

    methods
        %constructor
        function this = SupervisedHebbianLayer(size, transfer)
            if(nargin ~= 2)
                error("Invalid number of arguments.")
            else
                this.numInputs = size;
                this.weights = rand(this.numInputs, this.numInputs);
                this.transferFunc = transfer;
            end
        end

        %train function
        function this = train(this, target, pattern)
            this.weights = this.weights + target * pattern';
        end

         %training using the pseudoinverse rule
        function this = pseudoInverseRule(this, input, target)
            this.weights = target * pinv(input);
            % Define the threshold for rounding
            threshold = 1e-6; % Adjust this threshold as needed
            % Round the elements of the weights matrix to integers if they're close enough
            [rows, cols] = size(this.weights);
            for i = 1:rows
                for j = 1:cols
                    if abs(this.weights(i, j) - round(this.weights(i, j))) < threshold
                        this.weights(i, j) = round(this.weights(i, j));
                    end
                end
            end
        end

        %% --- forward ---
        %factory
        function func = doFunc(this, n)
            switch this.transferFunc
                case "hardlim"
                    func = this.hardlim(n);
                case "hardlims"
                    func = this.hardlims(n);
                case "purelin"
                    func = this.purelin(n);
                otherwise
                    error("Transfer function not supported");
            end
        end

        %hardlim
        function f = hardlim(this, n)
            if(n < 0)
                f = 0;
            else %if n >= 0
                f = 1;
            end
        end

        %hardlimS
        function f = hardlims(this, n)
            if(n < 0)
                f = -1;
            else % n >=0
                f = 1;
            end
        end

        %purelin
        function f = purelin(this, n)
            f = n;
        end

        %do foward
         function output = forward(this, input)
            n = (this.weights * input');
            output = arrayfun(@this.doFunc, n);
        end

    end

    methods (Static)
        %%addNoise to a vector, distort it 
        function pvec = addNoise(pvec, num)
            % ADDNOISE Add noise to "binary" vector
            % pvec pattern vector (-1 and 1)
            % num number of elements to flip randomly
            % Handle special case where there's no noise
            if num == 0
                return;
            end
            % first, generate a random permutation of all indices into pvec
            inds = randperm(length(pvec));
            % then, use the first n elements to flip pixels
            pvec(inds(1:num)) = -pvec(inds(1:num));
        end 

        %calculate the erros of neuron given the target value and produced output
        function e = errorLoss(a, t)
            e = t - a;
        end

        %----different printing functions----%
        %print the image out to a color map, use for weights (output) because of
        %color scale
        function printNumbs(vec, fig_num, name)
            %adjust to be printed, resshaped for matrix, inverted for grey scale
            figure(fig_num);
            matrix = 1 - rot90(flipud(reshape(vec, 5, 6)), 3);
            % Display the matrix using imagesc
            imshow(matrix, 'InitialMagnification', 'fit', 'Colormap', gray);
            title('Number Visualization: ' + name);
        end
        
        % Print to console, can be used for output or input
        function printWeights(vec, fig_num, name)
            figure(fig_num);
            imagesc(vec);
            colormap(hsv);
            colorbar;
            title('Weight Visualization: ' + name);
        end
    end
end