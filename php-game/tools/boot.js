const originalHTML = document.documentElement.outerHTML;
try {
  if (!crypto.subtle || !window.Worker || !window.DecompressionStream) throw new Error('Use a current browser over HTTPS (or localhost for a local preview).');
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(originalHTML));
  const release = Array.from(new Uint8Array(digest), v => v.toString(16).padStart(2, '0')).join('');
  const base = new URL('./', location.href).href;
  const style = document.createElement('link'); style.rel = 'stylesheet'; style.href = new URL('assets/style.css?v=' + release, base); document.head.append(style);
  const app = await import(new URL('assets/game.js?v=' + release, base));
  await app.start({ base, release });
} catch (error) {
  document.getElementById('status').textContent = 'Astrosynapse could not start.';
  const box = document.getElementById('error'); box.textContent = error.message; box.hidden = false;
}
