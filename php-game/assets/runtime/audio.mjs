import { resource } from './resources.mjs';
export const SOUNDS = ['attack', 'authority', 'combat', 'playerturn', 'scrap', 'shuffle', 'trade'];

// Buffers are decoded once; every playback gets its own overlapping source.
export class GameAudio {
  constructor({ Context = globalThis.AudioContext || globalThis.webkitAudioContext, load = resource,
    later = setTimeout, cancel = clearTimeout } = {}) {
    Object.assign(this, { load, later, cancel, buffers: new Map(), queue: [], sources: new Set(), lastStart: -Infinity, timer: null });
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
  enqueue(names) { this.queue.push(...names); this.pump(); }
  pump() {
    if (!this.ready || this.context?.state !== 'running' || this.timer !== null) return;
    while (this.queue.length && !this.buffers.has(this.queue[0])) this.queue.shift();
    if (!this.queue.length) return;
    const remaining = this.lastStart + 250 - this.context.currentTime * 1000;
    if (remaining > 0) {
      this.timer = this.later(() => { this.timer = null; this.pump(); }, Math.ceil(remaining));
      return;
    }
    const source = this.context.createBufferSource();
    source.buffer = this.buffers.get(this.queue.shift());
    source.connect(this.gain);
    source.onended = () => { source.disconnect(); this.sources.delete(source); };
    this.sources.add(source);
    source.start();
    this.lastStart = this.context.currentTime * 1000;
    this.pump();
  }
  clear() {
    this.cancel(this.timer); this.timer = null; this.queue.length = 0;
    for (const source of this.sources) source.stop();
    this.sources.clear();
  }
}
