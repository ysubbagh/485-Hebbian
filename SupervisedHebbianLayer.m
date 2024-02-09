classdef SupervisedHebbianLayer
    properties
        weights
        inputSize %also output size
        learningRate
        transferFunction
    end

    methods
        %%constructor
        function this = SupervisedHebbianLayer(p1, p2)
            if(isscalar(p1)) %input size passed
                this.inputSize = p1;
                %make weights 0 based on size
                this.weights = randn(this.inputSize,1);
            else %weights passed
                this.weights = p1;
                this.inputSize = size(this.weights, 1);
            end
            %simplify have learning rule = 1
            this.learningRate = 1;
            this.transferFunction = p2;
        end

        %%forward function, compute the autoassociator
        function output = forward(this, input)
            output = zeros(this.inputSize, 1);
            for i = 1:this.inputSize
                n = this.weights(i, :) * input;
                output(i) = this.doFunc(n);
            end
        end

        %%training functions
        %one at a time
        function this = train(this, input, target)
            input = input(:);
            target = target(:);
            this.weights = this.weights + this.learningRate * (target' * input);
        end

        %training using the pseudoinverse rule
        function this = pseudoInverseRule(this, input, target)
            this.weights = target * pinv(input);
            %becuase reformatting weirdly
            this.weights = this.weights(:);
            %because of negtaive zero
            this.weights(abs(this.weights) < eps) = 0;
            %this.weights = target * (inv(input' * input) * input');
        end

        %---transfer functions---%
        %transfer function FACTORY
        function func = doFunc(this, n)
            switch this.transferFunction
                case "hardlim"
                    func = this.hardlim(n);
                case "hardlims"
                    func = this.hardlims(n);
                case "purelin"
                    func = this.purelin(n);
                otherwise
                    error("Transfer function not supported.");
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
            else %% n >=0
                f = 1;
            end
        end
        %purelins
        function f = purelin(this, n)
            f = n;
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