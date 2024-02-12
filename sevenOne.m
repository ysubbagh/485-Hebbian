%initalize the layer
network = SupervisedHebbianLayer(4, "hardlims");

%input patterns
p1 = [-1 1 -1 1];
p2 = [1 -1 1 1];

%test pattern
pt = [1 1 1 1];

%training the network
network = network.train(p1', p1');
network = network.train(p2', p2');

%test network
output = network.forward(pt);

%print for validation
disp("weights: ");
disp(network.weights);

disp("output: ");
disp(output);


%are the patterns orthogonal?
dotProd = dot(p1, p2);
disp("dot product = " + dotProd);
if(dotProd == 0)
    disp("Patterns ARE orthogonal!");
else
    disp("Patterns are NOT orthogonal!");
end

