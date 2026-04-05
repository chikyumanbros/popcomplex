let _state = 0x12345678 >>> 0;
let _seed = _state;

export function setRandomSeed(seed: number) {
  _seed = seed >>> 0;
  _state = _seed === 0 ? 0x9e3779b9 : _seed;
}

export function getRandomSeed(): number {
  return _seed >>> 0;
}

// xorshift32: deterministic, fast, adequate for simulation stochasticity.
export function randomF32(): number {
  let x = _state >>> 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  _state = x >>> 0;
  return (_state >>> 0) / 4294967296;
}

export function randomInt(maxExclusive: number): number {
  if (maxExclusive <= 0) return 0;
  return Math.floor(randomF32() * maxExclusive);
}
