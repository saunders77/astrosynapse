const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');

for (const file of ['tools/index.template.html', 'index.html']) {
  test(`${file}: UI initializes without a model management button`, () => {
    const html = fs.readFileSync(`${__dirname}/../${file}`, 'utf8');
    const nodes = [...html.matchAll(/<[^!/>][^>]*>/g)].map(([tag]) => ({
      id: tag.match(/\bid="([^"]+)"/)?.[1],
      dataset: Object.fromEntries([...tag.matchAll(/data-([\w-]+)="([^"]+)"/g)].map(([, key, value]) => [key, value])),
      listeners: {},
      addEventListener(event, callback) { this.listeners[event] = callback; },
      showModal() { this.open = true; },
      close() { this.open = false; },
    }));
    const document = {
      getElementById: id => nodes.find(node => node.id === id) || null,
      querySelectorAll: selector => nodes.filter(node => selector.slice(6, -1) in node.dataset),
    };
    const context = vm.createContext({
      document, location: { pathname: '/' }, localStorage: { getItem: () => null },
      IntersectionObserver: class {}, ResizeObserver: class { observe(node) { assert.ok(node); } },
    });
    const source = fs.readFileSync(`${__dirname}/../assets/runtime/ui.mjs`, 'utf8')
      .replace(/^import .*;$/gm, '').replace(/^export /gm, '');
    vm.runInContext(source, context);
    assert.equal(document.getElementById('models-open'), null);
  });
}
