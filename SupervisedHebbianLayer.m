classdef SupervisedHebbianLayer
    properties
        weights
        inputSize
        learningRate
    end

    methods
        %%constructor
        function this = SupervisedHebbianLayer(inputSize, learningRate)
            this.inputSize = inputSize;
            this.learningRate = learningRate;
            %make weights 0 based on size
            this.weights = zeros(inputSize);
        end

        %%forward function, compute the autoassociator
        function output = forward(this, inputPattern)
            output = this.weights * inputPattern';
        end

        %%training functions
        %one at a time
        function this = trainIndividually(this, input, target)
            weightAdjustment = this.learningRate * (target - this.forward(input)) * input;
            this.weights = this.weights + weightAdjustment;
        end

        %all at once, using psuedoinverse
        function this = trainAtOnce(this, input, target)
            this.weights = target' * pinv(input');
        end

    end

    %%methods outside of the class
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

    end


end