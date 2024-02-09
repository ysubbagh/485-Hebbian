%initalize the layer
network = SupervisedHebbianLayer(4, "hardlim");

%input patterns
p1 = [0 1 0 1];
p2 = [1 0 1 1];
p = [p1; p2];

%test pattern
pt = [1 1 1 1];

%training the network
network = network.train(p1, p1);
network = network.train(p2, p2);

%test network
output = network.forward(pt);

%print for validation
disp("weights: ");
disp(network.weights);

disp("output: ");
disp(output);


