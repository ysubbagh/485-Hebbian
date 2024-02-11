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
            %becuase reformatting weirdly
            this.weights = this.weights(:);
            %because of negtaive zero
            this.weights(abs(this.weights) < eps) = 0;
            %this.weights = target * (inv(input' * input) * input');
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
            else %% n >=0
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
end