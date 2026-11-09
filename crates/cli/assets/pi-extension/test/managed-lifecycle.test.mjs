// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { managedWiring } from '../managed.ts';
import { createExtension } from '../src/extension.ts';
import { createSharedLease } from '../src/managed/mcp-lease.ts';

const CREDENTIAL = Buffer.alloc(32, 7).toString('base64url');
const DAEMON = 'https://relay.example.com';

const spawnWithNode = (command, args, options) =>
  spawn(process.execPath, [command, ...args], options);

function fakePi() {
  const handlers = new Map();
  const registrations = [];
  return {
    pi: {
      on(name, handler) {
        handlers.set(name, handler);
      },
      registerProvider(name, config) {
        registrations.push({ name, config });
      },
    },
    fire: (name, event, context) => handlers.get(name)({ type: name, ...event }, context),
    registrations,
  };
}

function countingLease() {
  const counts = { ensureReady: 0, releases: 0 };
  return {
    counts,
    lease: {
      ensureReady: async () => {
        counts.ensureReady += 1;
      },
      release: async () => {
        counts.releases += 1;
      },
    },
  };
}

async function withRecordedFetch(run) {
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, request) => {
    requests.push({ url: String(url), headers: request.headers, body: JSON.parse(request.body) });
    return new Response('{}', { status: 200 });
  };
  try {
    return await run(requests);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

describe('managed pi MCP lease', () => {
  it('becomes ready, restarts, and releases without another launch', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'nemo-relay-pi-lease-'));
    const launches = join(directory, 'launches');
    const dispatcher = join(directory, 'dispatcher.mjs');
    await writeFile(
      dispatcher,
      String.raw`import { appendFileSync, readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
appendFileSync(${JSON.stringify(launches)}, 'launch\n');
const launchCount = readFileSync(${JSON.stringify(launches)}, 'utf8').trim().split('\n').length;
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on('line', (line) => {
  const request = JSON.parse(line);
  if (request.method === 'initialize') {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: request.id, result: {
      protocolVersion: '2025-11-25', capabilities: {}, serverInfo: { name: 'nemo-relay', version: 'test' }
    } }) + '\n');
  } else if (request.method === 'notifications/initialized' && launchCount === 1) {
    setTimeout(() => process.exit(0), 10);
  }
});
`,
    );

    let removed = 0;
    const lease = createSharedLease(
      {
        schema: 'nemo-relay-managed-pi-v2',
        daemonAddress: 'http://127.0.0.1:47632',
        dispatcherCommand: dispatcher,
      },
      CREDENTIAL,
      () => {
        removed += 1;
      },
      spawnWithNode,
    );
    const launchCount = async () => (await readFile(launches, 'utf8')).trim().split('\n').length;

    await lease.ensureReady();
    const deadline = Date.now() + 10_000;
    while ((await launchCount()) < 2) {
      assert.ok(Date.now() < deadline, 'MCP child did not restart');
      await new Promise((resolve) => setTimeout(resolve, 10));
      await lease.ensureReady();
    }
    await lease.release();
    assert.equal(await launchCount(), 2);
    assert.equal(removed, 1);
    await assert.rejects(lease.ensureReady(), /released/);
    assert.equal(await launchCount(), 2);
    await rm(directory, { recursive: true, force: true });
  });
});

describe('managed pi wiring', () => {
  const context = (model, catalog) => ({
    cwd: '/workspace',
    model,
    modelRegistry: { getAll: () => catalog },
    sessionManager: { getSessionId: () => 'session' },
  });

  it('keeps hook order across session replacement and releases the lease only on quit', async () => {
    const { pi, fire } = fakePi();
    const { lease, counts } = countingLease();
    const runtime = {
      config: {
        schema: 'nemo-relay-managed-pi-v2',
        daemonAddress: DAEMON,
        dispatcherCommand: '/opt/nemo-relay',
      },
      credential: CREDENTIAL,
      lease,
    };
    await withRecordedFetch(async (requests) => {
      createExtension(pi, { onFault: 'closed', bind: async () => managedWiring(runtime) });
      const ctx = context(undefined, []);
      await fire('session_start', { reason: 'startup' }, ctx);
      await Promise.all([
        fire('turn_start', { turnIndex: 0, timestamp: 1 }, ctx),
        fire('turn_start', { turnIndex: 1, timestamp: 2 }, ctx),
      ]);
      await fire('session_shutdown', { reason: 'reload' }, ctx);
      await fire('session_shutdown', { reason: 'new' }, ctx);
      const beforeQuit = counts.ensureReady;
      await fire('session_shutdown', { reason: 'quit' }, ctx);

      assert.deepEqual(
        requests.map((request) => request.body.hook_event_name),
        ['session_start', 'turn_start', 'turn_start', 'session_shutdown', 'session_shutdown'],
      );
      assert.deepEqual(
        requests
          .filter((request) => request.body.hook_event_name === 'turn_start')
          .map((request) => request.body.turn_index),
        [0, 1],
      );
      for (const request of requests) {
        assert.equal(request.url, `${DAEMON}/hooks/pi`);
        assert.equal(request.headers['x-nemo-relay-client-token'], CREDENTIAL);
        assert.equal(request.body.session_id, 'session');
      }
      assert.equal(
        counts.ensureReady - beforeQuit,
        0,
        'quit posts its final hook without restarting MCP',
      );
      assert.equal(counts.releases, 1);
    });
  });

  it('redirects the selected provider through the daemon, naming its endpoint', async () => {
    const { pi, fire, registrations } = fakePi();
    const { lease } = countingLease();
    const runtime = {
      config: {
        schema: 'nemo-relay-managed-pi-v2',
        daemonAddress: DAEMON,
        dispatcherCommand: '/opt/nemo-relay',
      },
      credential: CREDENTIAL,
      lease,
    };
    const selected = {
      id: 'custom-response-model',
      api: 'openai-responses',
      provider: 'custom-enterprise',
      baseUrl: 'https://llm.example/v1',
    };
    const sibling = {
      id: 'custom-messages-model',
      api: 'anthropic-messages',
      provider: 'custom-enterprise',
      baseUrl: 'https://llm.example/v1/',
    };
    await withRecordedFetch(async (requests) => {
      createExtension(pi, { onFault: 'closed', bind: async () => managedWiring(runtime) });
      const ctx = context(selected, [selected, sibling]);
      await fire('session_start', { reason: 'startup' }, ctx);
      await fire('session_shutdown', { reason: 'quit' }, ctx);

      assert.deepEqual(registrations, [
        {
          name: 'custom-enterprise',
          config: {
            baseUrl: DAEMON,
            headers: {
              'x-nemo-relay-client-token': CREDENTIAL,
              'x-nemo-relay-upstream-base-url': 'https://llm.example/v1',
            },
          },
        },
      ]);
      const [redirect] = requests.filter(
        (request) => request.body.hook_event_name === 'model_redirect',
      );
      assert.equal(redirect.body.outcome, 'redirect');
      assert.equal(redirect.body.provider, 'custom-enterprise');
      assert.equal(redirect.body.upstream, 'https://llm.example/v1');
    });
  });

  it('fails closed with the configuration error when the runtime cannot bind', async () => {
    const { pi, fire } = fakePi();
    createExtension(pi, {
      onFault: 'closed',
      bind: async () => {
        throw new Error('managed pi configuration is unreadable: ENOENT');
      },
    });
    const ctx = context(undefined, []);
    const result = await fire(
      'tool_call',
      { toolCallId: 'c1', toolName: 'read', input: { path: 'README.md' } },
      ctx,
    );
    assert.equal(result.block, true);
    assert.match(result.reason, /could not load its configuration/);
    assert.match(result.reason, /ENOENT/);
  });
});
