%initalize the layer
network = SupervisedHebbianLayer(4, "hardlim");

%patterns
p1 = [0 1 0 1];
p2 = [1 0 1 1];
p = [p1; p2];

%target
pt = [1 1 1 1];

disp("weights:");
disp(network.weights);

output = network.forward(p1);

disp("output:");
disp(output);

network = network.train(output, p1);

disp("weights:");
disp(network.weights);



