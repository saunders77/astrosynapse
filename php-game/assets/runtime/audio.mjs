import { resource } from './resources.mjs';
export const SOUNDS = ['win', 'lose', 'acquire', 'attack', 'authority', 'combat', 'playerturn', 'scrap', 'shuffle', 'trade'];

// Buffers are decoded once; sound starts and actions share a 250 ms queue cadence.
// Clips may overlap: their duration never adds another wait.
export class GameAudio {
  constructor({ Context = globalThis.AudioContext || globalThis.webkitAudioContext, load = resource,
    later = globalThis.setTimeout.bind(globalThis), cancel = globalThis.clearTimeout.bind(globalThis) } = {}) {
    Object.assign(this, { load, later, cancel, buffers: new Map(), queue: [], sources: new Set(), timer: null });
    try {
      this.context = new Context();
      this.gain = this.context.createGain();
      this.gain.connect(this.context.destination);
      this.context.addEventListener('statechange', () => this.pump());
    } catch { /* Audio support must not prevent playing the game. */ }
  }
  preload() {
    return this.loading ??= Promise.allSettled(SOUNDS.map(async name => {
      if (!this.context) return;
      const bytes = await this.load(`assets/audio/${name}.mp3`);
      this.buffers.set(name, await this.context.decodeAudioData(bytes.slice(0)));
    })).then(() => { this.ready = true; this.pump(); });
  }
  unlock() {
    if (this.context && this.context.state !== 'running') this.context.resume().then(() => this.pump()).catch(() => {});
  }
  setVolume(value) {
    if (this.gain) this.gain.gain.setValueAtTime(Math.max(0, Math.min(1, value)), this.context.currentTime);
  }
  enqueue(names, onStart = () => {}) {
    return new Promise(resolve => {
      this.queue.push({ names: [...names], onStart, resolve, started: false });
      this.pump();
    });
  }
  pump() {
    if (!this.ready || this.timer !== null || !this.queue.length) return;
    const item = this.queue[0];
    const name = item.names.shift();
    // Missing files, unsupported audio and blocked autoplay must not lock the UI.
    if (this.context?.state === 'running' && this.buffers.has(name)) {
      const source = this.context.createBufferSource();
      source.buffer = this.buffers.get(name);
      source.connect(this.gain);
      source.onended = () => { source.disconnect(); this.sources.delete(source); };
      this.sources.add(source);
      try { source.start(); }
      catch { source.disconnect(); this.sources.delete(source); }
    }
    if (!item.started) { item.started = true; item.onStart(); }
    this.timer = this.later(() => {
      this.timer = null;
      if (!item.names.length) { this.queue.shift(); item.resolve(); }
      this.pump();
    }, 250);
  }
  clear() {
    this.cancel(this.timer); this.timer = null; for (const item of this.queue.splice(0)) item.resolve();
    for (const source of this.sources) source.stop();
    this.sources.clear();
  }
}
