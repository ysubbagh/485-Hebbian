%initalize the layer
network = SupervisedHebbianLayer(4);

%patterns
p1 = [0 1 0 1];
p2 = [1 0 1 1];
p = [p1; p2];

%target
pt = [1 1 1 1];

disp("weights:" + network.weights);

network = network.train(p, pt);

disp("weights:" + network.weights);

