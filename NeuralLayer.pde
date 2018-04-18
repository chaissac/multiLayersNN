class NeuralLayer {
  int inputNodes, outputNodes ;
  PMatrix weights, bias ;
  double learningRate ;
  String activationFunction ;
  final String version = "0.1";

  NeuralLayer(int input, int output, String fn) {

    inputNodes = input ;
    outputNodes = output ;

    weights = new PMatrix(outputNodes, inputNodes);
    weights.randomize();

    bias = new PMatrix(outputNodes, 1);
    bias.randomize();

    setLearningRate(0.5);
    setActivationFunction(fn);
  }

  NeuralLayer(NeuralLayer n) {

    inputNodes = n.inputNodes ;
    outputNodes = n.outputNodes ;

    weights = n.weights.clone();

    bias = n.bias.clone();

    learningRate = n.learningRate;
    setActivationFunction(n.activationFunction);
  }
  void setLearningRate(Double l) {
    learningRate = learningRate;
  }

  void setLearningRate(double lr) {
    learningRate = lr;
  }   
  void setLearningRate(float lr) {
    learningRate = (double)lr;
  }   
  void setLearningRate(int lr) {
    learningRate = (double)lr;
  } 
  String checkFunction(String fn) {
    fn.toLowerCase();
    switch (fn) {
    case "sigmoid":
    case "tanh":
    case "step":
    case "relu":
      return fn;
    default:
      return "sigmoid";
    }
  }
  void setActivationFunction(String func) {
    activationFunction = checkFunction(func);
  }  
  String toJson() {
    JSONObject json = new JSONObject();
    json.put( "class", "NeuralLayer" );
    json.put( "version", version );
    json.put( "inputNodes", inputNodes );
    json.put( "outputNodes", outputNodes );
    json.put( "learningRate", learningRate );
    json.put( "activationFunction", activationFunction );
    json.put( "weights", weights.toJson() );
    json.put( "bias", bias.toJson() );
    return json.toString();
  }  
  void save(String file) {
    String json = toJson();
    println(json);
  }
  Double[] run(PMatrix inputs) {
    PMatrix output = weights.clone();
    output.product(inputs);
    output.add(bias);
    // activation function!
    output.map(activationFunction);
    // Sending back to the caller!
    return output;
  }

  void train(PMatrix inputs, PMatrix targets) {

    // Generating the Outputs
    PMatrix outputs = weights.clone();
    outputs.product(inputs);
    outputs.add(bias);
    // activation function!
    outputs.map(activationFunction);

    // Calculate the error
    // ERROR = TARGETS - OUTPUTS
    PMatrix outputErrors = targets.clone();
    outputErrors.sub(outputs);

    // Calculate gradient
    PMatrix gradient = outputs.clone();
    gradient.map("d"+activationFunction);
    gradient.mult(outputErrors);
    gradient.mult(learningRate);

    // Calculate input->outpout deltas
    PMatrix inputsT = inputs.clone();
    inputsT.transpose();
    PMatrix weightDeltas = gradient.clone();
    weightDeltas.product(inputsT);

    // Adjust the weights by deltas
    weights.add(weightDeltas);
    // Adjust the bias by its deltas (which is just the gradients)
    bias.add(gradient);
  }
}