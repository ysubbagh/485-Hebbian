close all;
%% setup network
network = SupervisedHebbianLayer(30, "hardlims");
pseudoNet = SupervisedHebbianLayer(30, "hardlims");

%input patterns
p0 = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
p1 = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1];
p2 = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1];
p3 = [-1 1 1 1 -1 1 -1 -1 -1 1 -1 -1 1 1 1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
p4 = [1 -1 -1 -1 1 1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1];
p5 = [1 1 1 1 1 1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
p = [p0' p1' p2' p3' p4' p5'];


%% train the networks
%using hebb rule
network = network.train(p,p);

%pseudoinverse train the function
pseudoNet = pseudoNet.pseudoInverseRule(p, p);

%% print for validation
SupervisedHebbianLayer.printNumbs(p0, 1, "0");
SupervisedHebbianLayer.printNumbs(p1, 2, "1");
SupervisedHebbianLayer.printNumbs(p2, 3, "2");
SupervisedHebbianLayer.printNumbs(p3, 4, "3");
SupervisedHebbianLayer.printNumbs(p4, 5, "4");
SupervisedHebbianLayer.printNumbs(p5, 6, "5");
SupervisedHebbianLayer.printWeights(network.weights, 7, "traditional training");
SupervisedHebbianLayer.printWeights(pseudoNet.weights, 8, "pseudoinverse training");

%% adding noise
% Initialize accuracy matrix
noiseLevels = [2 4 6];
accuracyMatrixHebb = zeros(length(noiseLevels), 5);
accuracyMatrixPS = zeros(length(noiseLevels), 5);

% Loop over each noise level
for j = 1:length(noiseLevels)
    noiseLevel = noiseLevels(j);
    
    % Loop over each pattern
    for i = 1:length(accuracyMatrixHebb)
        % Setup network
        testHebb = SupervisedHebbianLayer(30, "hardlims");
        testPS = SupervisedHebbianLayer(30, "hardlims");
        
        % Setup training patterns
        if i == 1
            P = [p0' p1'];
        else
            nextPattern = getPattern(i);
            P = [P nextPattern'];
        end
        
        % Train the network
        testHebb = testHebb.train(P, P);
        testPS = testPS.train(P, P);
        
        % Initialize correct count for this noise level
        correctCountHebb = 0;
        correctCountPS = 0;
        
        % Test 100 versions
        for k = 1:numVersions
            % Create a noisy version
            if i == 1
                %hebb, setup noise and create output
                noisyInput0 = SupervisedHebbianLayer.addNoise(p0, noiseLevel);
                output0Hebb = testHebb.forward(noisyInput0);
                noisyInput1 = SupervisedHebbianLayer.addNoise(p1, noiseLevel);
                output1Hebb = testHebb.forward(noisyInput1);
                
                % Check for correctness in hebb
                if isCorrectlyClassified(output0Hebb, p0)
                    correctCountHebb = correctCountHebb + 1;
                end
                if isCorrectlyClassified(output1Hebb, p1)
                    correctCountHebb = correctCountHebb + 1;
                end

                %ps, setup noise and create output
                output0PS = testPS.forward(noisyInput0);
                output1PS = testPS.forward(noisyInput1);
                
                % Check for correctness in psuedoinverse
                if isCorrectlyClassified(output0PS, p0)
                    correctCountPS = correctCountPS + 1;
                end
                if isCorrectlyClassified(output1PS, p1)
                    correctCountPS = correctCountPS + 1;
                end

            else
                %hebb classify and check correctness
                noisyInput = SupervisedHebbianLayer.addNoise(nextPattern, noiseLevel);
                % Classify the noisy image
                outputHebb = testHebb.forward(noisyInput);
                outputPS = testPS.forward(noisyInput);
                
                % Check for correctness
                if isCorrectlyClassified(outputHebb, nextPattern)
                    correctCountHebb = correctCountHebb + 1;
                end

                %ps classift and check correctness
                if isCorrectlyClassified(outputPS, nextPattern)
                    correctCountPS = correctCountPS + 1;
                end
            end
        end
        
        % Calculate accuracy for this pattern and noise level
        %for hebb
        if i == 1
            correctCountHebb = correctCountHebb / 2;
        end
        accuracyMatrixHebb(j, i) = (correctCountHebb / numVersions) * 100;
        %for ps
        if i == 1
            correctCountPS = correctCountPS / 2;
        end
        accuracyMatrixPS(j, i) = (correctCountPS / numVersions) * 100;
    end
end


disp(accuracyMatrixHebb);
disp(accuracyMatrixPS);

%% print results
xTicks = [2, 3, 4, 5, 6]; % Define the x-axis ticks
% Plot the graph for hebb rule
figure; % Create a new figure for the plot
hold on; % Hold the plot to overlay multiple lines
for i = 1:size(accuracyMatrixHebb, 1) % Iterate over each row (each noise level)
    plot(xTicks, accuracyMatrixHebb(i, :), '-o', 'DisplayName', sprintf('Noise Level %d', xTicks(i)));
end
hold off; % Release the hold on the plot
xlabel('Number of Input Patterns Stored for Training');
ylabel('Classification Accuracy (%)');
title('Network Performance of Hebbian Rule with Noisy Inputs');
legend('show'); % Show the legend with noise level labels
grid on;
xticks(2:6); % Set x-axis ticks to integers from 2 to 6

% Plot the graph for pseudoinverse rule
figure; % Create a new figure for the plot
hold on; % Hold the plot to overlay multiple lines
for i = 1:size(accuracyMatrixPS, 1) % Iterate over each row (each noise level)
    plot(xTicks, accuracyMatrixPS(i, :), '-o', 'DisplayName', sprintf('Noise Level %d', xTicks(i)));
end
hold off; % Release the hold on the plot
xlabel('Number of Input Patterns Stored for Training');
ylabel('Classification Accuracy (%)');
title('Network Performance of Pseudoinverse Rule with Noisy Inputs');
legend('show'); % Show the legend with noise level labels
grid on;
xticks(2:6); % Set x-axis ticks to integers from 2 to 6

%% functions to aid
%for noise addition, get the input patterns
function pattern = getPattern(num)
    switch num
        case 0
            pattern = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
        case 1
            pattern = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1];
         case 2
            pattern = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1];
        case 3
            pattern = [-1 1 1 1 -1 1 -1 -1 -1 1 -1 -1 1 1 1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
        case 4
            pattern = [1 -1 -1 -1 1 1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1];
        case 5
            pattern = [1 1 1 1 1 1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
        otherwise
            error("pattern doesnt exist.");
    end
end

%check if correct
function correct = isCorrectlyClassified(output, target)
    [~, predictedClass] = max(output);
    [~, trueClass] = max(target);
    correct = predictedClass == trueClass;
end


