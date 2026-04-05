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

  forward(input: Float32Array): Float32Array {
    const hidden = new Float32Array(this.hiddenSize);
    for (let h = 0; h < this.hiddenSize; h++) {
      let sum = 0;
      for (let i = 0; i < this.inputSize; i++) {
        sum += (input[i] * this.inputGain[i]) * this.weightsIH[i * this.hiddenSize + h];
      }
      sum += this.biasH[h];
      hidden[h] = Math.tanh(sum);
    }

    const output = new Float32Array(this.outputSize);
    for (let o = 0; o < this.outputSize; o++) {
      let sum = 0;
      for (let h = 0; h < this.hiddenSize; h++) {
        sum += hidden[h] * this.weightsHO[h * this.outputSize + o];
      }
      sum += this.biasO[o];
      output[o] = sum;
    }

    // Softmax
    let maxVal = -Infinity;
    for (let i = 0; i < output.length; i++) if (output[i] > maxVal) maxVal = output[i];
    let expSum = 0;
    for (let i = 0; i < output.length; i++) {
      output[i] = Math.exp(output[i] - maxVal);
      expSum += output[i];
    }
    for (let i = 0; i < output.length; i++) output[i] /= expSum;

    return output;
  }
}
