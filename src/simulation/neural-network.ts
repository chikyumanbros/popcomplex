export class NeuralNetwork {
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  weightsIH: Float32Array;
  weightsHO: Float32Array;
  inputGain: Float32Array;
  biasH: Float32Array;
  biasO: Float32Array;

  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.weightsIH = new Float32Array(inputSize * hiddenSize);
    this.weightsHO = new Float32Array(hiddenSize * outputSize);
    this.inputGain = new Float32Array(inputSize).fill(1);
    this.biasH = new Float32Array(hiddenSize);
    this.biasO = new Float32Array(outputSize);
  }

  loadFromTapeWeights(weights: Float32Array) {
    const ihCount = this.inputSize * this.hiddenSize;
    const hoCount = this.hiddenSize * this.outputSize;
    let off = 0;

    for (let i = 0; i < ihCount; i++, off++) {
      this.weightsIH[i] = off < weights.length ? weights[off] : 0;
    }
    for (let i = 0; i < hoCount; i++, off++) {
      this.weightsHO[i] = off < weights.length ? weights[off] : 0;
    }
    for (let i = 0; i < this.inputSize; i++, off++) {
      this.inputGain[i] = off < weights.length ? weights[off] : 1;
    }
    for (let i = 0; i < this.hiddenSize; i++, off++) {
      this.biasH[i] = off < weights.length ? weights[off] : 0;
    }
    for (let i = 0; i < this.outputSize; i++, off++) {
      this.biasO[i] = off < weights.length ? weights[off] : 0;
    }
  }

  /**
   * Writes softmax mood probabilities into `moodOut` and post-tanh hidden activations into `primitiveOut`
   * (same vector the output layer reads; interpretable as learned primitive drives before they combine into mood).
   */
  forward(input: Float32Array, moodOut: Float32Array, primitiveOut: Float32Array): void {
    if (moodOut.length < this.outputSize || primitiveOut.length < this.hiddenSize) {
      throw new Error('NeuralNetwork.forward: moodOut / primitiveOut too small');
    }

    for (let h = 0; h < this.hiddenSize; h++) {
      let sum = 0;
      for (let i = 0; i < this.inputSize; i++) {
        sum += (input[i] * this.inputGain[i]) * this.weightsIH[i * this.hiddenSize + h];
      }
      sum += this.biasH[h];
      primitiveOut[h] = Math.tanh(sum);
    }

    // Match pre-refactor order: accumulate W·p first, then add bias (FP rounding differs if bias is folded in first).
    for (let o = 0; o < this.outputSize; o++) {
      let sum = 0;
      for (let h = 0; h < this.hiddenSize; h++) {
        sum += primitiveOut[h] * this.weightsHO[h * this.outputSize + o];
      }
      sum += this.biasO[o];
      moodOut[o] = sum;
    }

    let maxVal = -Infinity;
    for (let i = 0; i < this.outputSize; i++) if (moodOut[i] > maxVal) maxVal = moodOut[i];
    let expSum = 0;
    for (let i = 0; i < this.outputSize; i++) {
      moodOut[i] = Math.exp(moodOut[i] - maxVal);
      expSum += moodOut[i];
    }
    for (let i = 0; i < this.outputSize; i++) moodOut[i] /= expSum;
  }
}
