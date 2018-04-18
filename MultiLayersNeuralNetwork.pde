class MultiLayersNeuralNetwork {
  NeuralLayer[] multiLayers ;
  int nbLayers ;
  double learningRate ;
  String activationFunction ;
  final String version = "0.1";

  NeuralNetwork(int[] layers, String fn) {
    nbLayers = layers.length-1;
    multiLayers = new NeuralLayer[nbLayers];
    for(int i=0;i<nbLayers;i++) {
      multiLayers[i] = new NeuralNetwork(layers[i],layers[i+1],fn);
    }
    setLearningRate(0.5);
    setActivationFunction(fn);
  }
  void setLearningRate(Double lr) {
    learningRate = lr;
    for(NeuralNetwork n:multiLayers) n.setLearningRate(lr);
  }
  void setLearningRate(float lr) {
     setLearningRate((double)lr);
  }   
  void setLearningRate(int lr) {
    setLearningRate((double)lr);
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
    for(NeuralNetwork n:multiLayers) n.setActivationFunction(func);
  }  
  String toJson() {
    JSONObject json = new JSONObject();
    json.put( "class", "NeuralNetwork" );
    json.put( "version", version );
    json.put( "inputNodes", inputNodes );
    json.put( "hiddenNodes", hiddenNodes );
    json.put( "outputNodes", outputNodes );
    json.put( "learningRate", learningRate );
    json.put( "activationFunction", activationFunction );
    json.put( "weightsIH", weightsIH.toJson() );
    json.put( "weightsHO", weightsHO.toJson() );
    json.put( "biasH", biasH.toJson() );
    json.put( "biasO", biasO.toJson() );
    return json.toString();
  }  
  void save(String file) {
    String json = toJson();
    println(json);
  }
  Double[] predict(Double[] inputArray) {

    // Generating the Hidden Outputs
    PMatrix inputs = new PMatrix(inputArray);
    PMatrix hidden = weightsIH.clone();
    hidden.product(inputs);
    hidden.add(biasH);
    // activation function!
    hidden.map(activationFunction);

    // Generating the output's output!
    PMatrix output = weightsHO.clone() ;
    output.product(hidden);
    output.add(biasO);
    output.map(activationFunction);

    // Sending back to the caller!
    return output.toArray();
  }

  void train(Double[] inputArray, Double[] targetArray) {

    // Generating the Hidden Outputs
    PMatrix inputs = new PMatrix(inputArray);
    PMatrix hidden = weightsIH.clone();
    hidden.product(inputs);
    hidden.add(biasH);
    // activation function!
    hidden.map(activationFunction);

    // Generating the output's output!
    PMatrix outputs = weightsHO.clone() ;
    outputs.product(hidden);
    outputs.add(biasO);
    outputs.map(activationFunction);

    // Convert array to matrix object
    PMatrix targets = new PMatrix(targetArray);

    // Calculate the error
    // ERROR = TARGETS - OUTPUTS
    PMatrix outputErrors = targets.clone();
    outputErrors.sub(outputs);

    // let gradient = outputs * (1 - outputs);
    // Calculate gradient
    PMatrix gradient = outputs.clone();
    gradient.map("d"+activationFunction);
    gradient.mult(outputErrors);
    gradient.mult(learningRate);

    // Calculate deltas
    PMatrix hiddenT = hidden.clone();
    hiddenT.transpose();
    PMatrix weightHODeltas = gradient.clone();
    weightHODeltas.product(hiddenT);

    // Adjust the weights by deltas
    weightsHO.add(weightHODeltas);
    // Adjust the bias by its deltas (which is just the gradients)
    biasO.add(gradient);

    // Calculate the hidden layer errors
    PMatrix hiddenErrors = weightsHO.clone();
    hiddenErrors.transpose();
    hiddenErrors.product(outputErrors);

    // Calculate hidden gradient
    PMatrix hiddenGradient = hidden.clone();
    hiddenGradient.map("d"+activationFunction);
    hiddenGradient.mult(hiddenErrors);
    hiddenGradient.mult(learningRate);

    // Calculate input->hidden deltas
    PMatrix inputsT = inputs.clone();
    inputsT.transpose();
    PMatrix weightIHDeltas = hiddenGradient.clone();
    weightIHDeltas.product(inputsT);

    // Adjust the weights by deltas
    weightsIH.add(weightIHDeltas);
    // Adjust the bias by its deltas (which is just the gradients)
    biasH.add(hiddenGradient);
  }
}