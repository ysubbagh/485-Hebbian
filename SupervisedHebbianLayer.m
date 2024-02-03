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


end