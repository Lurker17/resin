const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.listeners = {};
    this.style = {};
    this.value = '';
    this.classes = new Set();
    this.classList = {
      add: name => this.classes.add(name),
      remove: name => this.classes.delete(name),
      contains: name => this.classes.has(name),
      toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name),
    };
  }
  set className(value) { this.classes = new Set(value.split(' ')); }
  get isConnected() { return this.root || Boolean(this.parent?.isConnected); }
  append(...children) { for (const child of children) { this.children.push(child); child.parent = this; } }
  before(child) { this.parent.children.splice(this.parent.children.indexOf(this), 0, child); child.parent = this.parent; }
  remove() { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); this.parent = null; }
  replaceChildren() { for (const child of [...this.children]) child.remove(); }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setAttribute() {}
  focus() {}
  getBoundingClientRect() { return {bottom: 0}; }
}

function demo(responses) {
  const nodes = [], elements = {}, timers = new Set(), requests = [];
  const body = new Element('body');
  body.root = true;
  const create = tag => { const node = new Element(tag); nodes.push(node); return node; };
  const context = vm.createContext({
    document: {
      body,
      documentElement: {scrollHeight: 0},
      createElement: create,
      querySelector: selector => {
        if (!elements[selector]) { elements[selector] = create(selector); body.append(elements[selector]); }
        return elements[selector];
      },
    },
    window: {innerHeight: 800, scrollY: 0, scrollTo() {}, addEventListener() {}},
    ResizeObserver: class { observe() {} },
    requestAnimationFrame: callback => callback(),
    performance: {now: () => 1000},
    setInterval: () => { const id = timers.size + 1; timers.add(id); return id; },
    clearInterval: id => timers.delete(id),
    TextDecoder, Uint8Array,
    fetch: async (url, options) => {
      if (url === '/v1/models') return {json: async () => ({data: [{id: 'text-only'}]})};
      requests.push(JSON.parse(options.body));
      const response = responses.shift();
      if (response.error) return {ok: false, json: async () => ({error: {message: response.error}})};
      let index = 0;
      return {ok: true, body: {getReader: () => ({read: async () => index < response.length
        ? {value: new TextEncoder().encode(response[index++]), done: false}
        : {done: true}})}};
    },
  });
  const html = fs.readFileSync(path.join(__dirname, '../crates/resin-http/src/index.html'), 'utf8');
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  elements['#temperature'].value = '1';
  elements['#top-p'].value = '1';
  elements['#max-output-tokens'].value = '100';
  return {
    elements, nodes, timers, requests, context,
    submit: async text => {
      elements['#prompt'].value = text;
      await elements['#form'].listeners.submit({preventDefault() {}});
    },
    checkReady: () => {
      assert.equal(elements['#send'].disabled, false);
      assert.equal(elements['#attach'].disabled, false);
      assert.equal(timers.size, 0);
      assert.equal(nodes.some(node => node.isConnected && node.classes.has('pending')), false);
    },
  };
}

const sse = value => `data: ${JSON.stringify(value)}\n\n`;
const done = 'data: [DONE]\n\n';

test('unsupported image SSE errors clear busy state and do not poison the next turn', async () => {
  const page = demo([
    [sse({error: {message: 'This endpoint supports text content only'}}), done],
    [sse({choices: [{delta: {content: 'Hello'}}]}), done],
  ]);
  vm.runInContext("pendingImages=[{name:'test.png',size:1,url:'data:image/png;base64,AA=='}]", page.context);
  await page.submit('What is in this image?');
  assert.equal(page.nodes.some(node => node.textContent === 'Error: This endpoint supports text content only'), true);
  assert.equal(vm.runInContext('history.length', page.context), 0);
  page.checkReady();
  await page.submit('Hello');
  assert.deepEqual(page.requests[1].messages, [{role: 'user', content: 'Hello'}]);
  assert.equal(vm.runInContext('history[1].content', page.context), 'Hello');
  page.checkReady();
});

test('premature EOF reports failure and clears the pending indicator', async () => {
  const page = demo([[]]);
  await page.submit('Hello');
  assert.equal(page.nodes.some(node => node.textContent === 'Error: The inference stream ended before completion'), true);
  assert.equal(vm.runInContext('history.length', page.context), 0);
  page.checkReady();
});

test('HTTP admission errors restore the composer and conversation', async () => {
  const page = demo([{error: 'Image request is too large'}]);
  await page.submit('Hello');
  assert.equal(page.nodes.some(node => node.textContent === 'Error: Image request is too large'), true);
  assert.equal(vm.runInContext('history.length', page.context), 0);
  page.checkReady();
});
