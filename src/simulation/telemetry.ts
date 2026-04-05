export interface TelemetrySnapshot {
  tapeCorruptions: number;
  writeRandomizations: number;
  channelBitflips: number;
  channelSwapsAccepted: number;
  channelSwapsRejected: number;
  readMisfetches: number;
  ruleDuplications: number;
  ruleSwaps: number;
  dataDuplications: number;
  reproductionAttempts: number;
  reproductionSuccess: number;
  birthsFromReproduce: number;
  birthsFromSplit: number;
  stillbirths: number;
  reproduceFailDominance: number;
  reproduceFailCrowding: number;
  splitEvents: number;
  splitFragments: number;
  splitFragmentCells: number;
  splitFragmentSingletons: number;
  splitLargestKeptCells: number;
  xenoTransferAttempts: number;
  xenoTransferSuccess: number;
  xenoTransferDriveSum: number;
  socialCohesionSamples: number;
  socialCohesionSum: number;
  actionExec: number[];
}

const ACTION_SLOTS = 16;

const counters: TelemetrySnapshot = {
  tapeCorruptions: 0,
  writeRandomizations: 0,
  channelBitflips: 0,
  channelSwapsAccepted: 0,
  channelSwapsRejected: 0,
  readMisfetches: 0,
  ruleDuplications: 0,
  ruleSwaps: 0,
  dataDuplications: 0,
  reproductionAttempts: 0,
  reproductionSuccess: 0,
  birthsFromReproduce: 0,
  birthsFromSplit: 0,
  stillbirths: 0,
  reproduceFailDominance: 0,
  reproduceFailCrowding: 0,
  splitEvents: 0,
  splitFragments: 0,
  splitFragmentCells: 0,
  splitFragmentSingletons: 0,
  splitLargestKeptCells: 0,
  xenoTransferAttempts: 0,
  xenoTransferSuccess: 0,
  xenoTransferDriveSum: 0,
  socialCohesionSamples: 0,
  socialCohesionSum: 0,
  actionExec: new Array(ACTION_SLOTS).fill(0),
};

function bump<K extends keyof TelemetrySnapshot>(key: K, amount = 1) {
  if (key === 'actionExec') return;
  counters[key] = (counters[key] as number) + amount as TelemetrySnapshot[K];
}

export function recordTapeCorruption() {
  bump('tapeCorruptions');
}
export function recordWriteRandomization() {
  bump('writeRandomizations');
}
export function recordChannelBitflip() {
  bump('channelBitflips');
}
export function recordChannelSwap(accepted: boolean) {
  bump(accepted ? 'channelSwapsAccepted' : 'channelSwapsRejected');
}
export function recordReadMisfetch() {
  bump('readMisfetches');
}
export function recordRuleDuplication() {
  bump('ruleDuplications');
}
export function recordRuleSwap() {
  bump('ruleSwaps');
}
export function recordDataDuplication() {
  bump('dataDuplications');
}
export function recordReproductionAttempt() {
  bump('reproductionAttempts');
}
export function recordReproductionSuccess() {
  bump('reproductionSuccess');
}
export function recordBirthFromReproduce() {
  bump('birthsFromReproduce');
}
export function recordSplitEvent(largestKeptCells: number) {
  bump('splitEvents');
  if (largestKeptCells > 0) bump('splitLargestKeptCells', largestKeptCells);
}
export function recordBirthFromSplit(fragmentCells: number) {
  bump('birthsFromSplit');
  bump('splitFragments');
  if (fragmentCells > 0) bump('splitFragmentCells', fragmentCells);
  if (fragmentCells === 1) bump('splitFragmentSingletons');
}
export function recordStillbirth() {
  bump('stillbirths');
}
export function recordReproduceFailDominance() {
  bump('reproduceFailDominance');
}
export function recordReproduceFailCrowding() {
  bump('reproduceFailCrowding');
}
export function recordXenoTransferAttempt(success: boolean, drive: number) {
  bump('xenoTransferAttempts');
  bump('xenoTransferDriveSum', Math.max(0, drive));
  if (success) bump('xenoTransferSuccess');
}
export function recordSocialCohesion(cohesion01: number) {
  bump('socialCohesionSamples');
  bump('socialCohesionSum', Math.max(0, Math.min(1, cohesion01)));
}
export function recordActionExecution(op: number) {
  if (op < 0 || op >= counters.actionExec.length) return;
  counters.actionExec[op]++;
}

export function snapshotAndResetTelemetry(): TelemetrySnapshot {
  const snap: TelemetrySnapshot = {
    tapeCorruptions: counters.tapeCorruptions,
    writeRandomizations: counters.writeRandomizations,
    channelBitflips: counters.channelBitflips,
    channelSwapsAccepted: counters.channelSwapsAccepted,
    channelSwapsRejected: counters.channelSwapsRejected,
    readMisfetches: counters.readMisfetches,
    ruleDuplications: counters.ruleDuplications,
    ruleSwaps: counters.ruleSwaps,
    dataDuplications: counters.dataDuplications,
    reproductionAttempts: counters.reproductionAttempts,
    reproductionSuccess: counters.reproductionSuccess,
    birthsFromReproduce: counters.birthsFromReproduce,
    birthsFromSplit: counters.birthsFromSplit,
    stillbirths: counters.stillbirths,
    reproduceFailDominance: counters.reproduceFailDominance,
    reproduceFailCrowding: counters.reproduceFailCrowding,
    splitEvents: counters.splitEvents,
    splitFragments: counters.splitFragments,
    splitFragmentCells: counters.splitFragmentCells,
    splitFragmentSingletons: counters.splitFragmentSingletons,
    splitLargestKeptCells: counters.splitLargestKeptCells,
    xenoTransferAttempts: counters.xenoTransferAttempts,
    xenoTransferSuccess: counters.xenoTransferSuccess,
    xenoTransferDriveSum: counters.xenoTransferDriveSum,
    socialCohesionSamples: counters.socialCohesionSamples,
    socialCohesionSum: counters.socialCohesionSum,
    actionExec: [...counters.actionExec],
  };

  counters.tapeCorruptions = 0;
  counters.writeRandomizations = 0;
  counters.channelBitflips = 0;
  counters.channelSwapsAccepted = 0;
  counters.channelSwapsRejected = 0;
  counters.readMisfetches = 0;
  counters.ruleDuplications = 0;
  counters.ruleSwaps = 0;
  counters.dataDuplications = 0;
  counters.reproductionAttempts = 0;
  counters.reproductionSuccess = 0;
  counters.birthsFromReproduce = 0;
  counters.birthsFromSplit = 0;
  counters.stillbirths = 0;
  counters.reproduceFailDominance = 0;
  counters.reproduceFailCrowding = 0;
  counters.splitEvents = 0;
  counters.splitFragments = 0;
  counters.splitFragmentCells = 0;
  counters.splitFragmentSingletons = 0;
  counters.splitLargestKeptCells = 0;
  counters.xenoTransferAttempts = 0;
  counters.xenoTransferSuccess = 0;
  counters.xenoTransferDriveSum = 0;
  counters.socialCohesionSamples = 0;
  counters.socialCohesionSum = 0;
  counters.actionExec.fill(0);
  return snap;
}
