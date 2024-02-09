%initalize the layer
network = SupervisedHebbianLayer(4, "hardlims");

%input patterns
p1 = [-1 1 -1 1];
p2 = [1 -1 1 1];
p = [p1; p2];

%test pattern
pt = [1 1 1 1];

%training the network
network = network.pseudoInverseRule(p, p);

%test network
output = network.forward(pt);

%print for validation
disp("weights: ");
disp(network.weights);

disp("output: ");
disp(output);


