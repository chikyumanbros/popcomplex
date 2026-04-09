import { type Tape, FEEDBACK_SLOTS } from './tape';
import { NeuralNetwork } from './neural-network';

export const NN_INPUT = 8;
export const NN_HIDDEN = 4;
export const NN_OUTPUT = 4;

/** One label per hidden tanh axis; semantics are evolved (not fixed “fear vs explore”). */
export const NN_PRIMITIVE_LABELS: readonly string[] = Object.freeze(
  Array.from({ length: NN_HIDDEN }, (_, i) => `p${i}`),
);

// NN output indices → organism "mood"
export const NN_EAT = 0;      // urgency to feed
export const NN_GROW = 1;     // urgency to grow/reproduce
export const NN_MOVE = 2;     // urgency to move
export const NN_CONSERVE = 3; // urgency to conserve/digest

export interface Organism {
  id: number;
  parentId: number | null;
  birthTick: number;
  tape: Tape;
  cells: Set<number>;
  age: number;
  nn: NeuralNetwork;
  nnInput: Float32Array;   // computed from world state each tick
  nnOutput: Float32Array;  // softmax output (4 "mood" probabilities)
  /** Post-tanh hidden layer (= primitive drives combined linearly into mood logits). Length NN_HIDDEN. */
  nnPrimitives: Float32Array;
  /** Instant stress proxy (0..1) mixed into NN inputs (reactive coupling to simulation stressors). */
  nnStress01: number;
  /** Instant territorial-claim proxy (0..1) used to gate NN input sensitivity. */
  nnClaim01: number;
  nnDominant: number;      // argmax of nnOutput (0-3)
  feedback: Uint8Array;
  reproduceCooldown: number; // ticks until REPRODUCE allowed again (anti-spam / monoculture brake)
}

export class OrganismManager {
  organisms: Map<number, Organism> = new Map();

  register(id: number, tape: Tape, opts?: { parentId?: number | null; birthTick?: number }) {
    const nn = new NeuralNetwork(NN_INPUT, NN_HIDDEN, NN_OUTPUT);
    nn.loadFromTapeWeights(tape.getNNWeights());

    this.organisms.set(id, {
      id,
      parentId: opts?.parentId ?? null,
      birthTick: opts?.birthTick ?? 0,
      tape,
      cells: new Set(),
      age: 0,
      nn,
      nnInput: new Float32Array(NN_INPUT),
      nnOutput: new Float32Array(NN_OUTPUT).fill(0.25),
      nnPrimitives: new Float32Array(NN_HIDDEN),
      nnStress01: 0,
      nnClaim01: 0,
      nnDominant: 0,
      feedback: new Uint8Array(FEEDBACK_SLOTS),
      reproduceCooldown: 0,
    });
  }

  get(id: number): Organism | undefined {
    return this.organisms.get(id);
  }

  remove(id: number) {
    this.organisms.delete(id);
  }

  tick() {
    for (const org of this.organisms.values()) {
      org.age++;
      // Integer ticks only; split-born cooldown used fractional crowd extra — floor avoids `--` going negative.
      if (org.reproduceCooldown > 0) {
        org.reproduceCooldown = Math.max(0, Math.floor(org.reproduceCooldown) - 1);
      }
    }
  }

  syncNeuralWeightsFromTape() {
    for (const org of this.organisms.values()) {
      org.nn.loadFromTapeWeights(org.tape.getNNWeights());
    }
  }

  get count(): number {
    return this.organisms.size;
  }
}
