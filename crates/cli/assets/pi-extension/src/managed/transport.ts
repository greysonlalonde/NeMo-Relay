// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type HookOutcome, errorDetail, outcomeFromResponse } from '../gateway.ts';

export const CLIENT_TOKEN_HEADER = 'x-nemo-relay-client-token';
export const UPSTREAM_BASE_URL_HEADER = 'x-nemo-relay-upstream-base-url';

const HOOK_TIMEOUT_MS = 30_000;
const MAX_HOOK_PAYLOAD_BYTES = 20 * 1024 * 1024;
const MAX_HOOK_RESPONSE_BYTES = 1024 * 1024;

export type DaemonEndpoint = {
  daemonAddress: string;
  credential: string;
};

class OversizedResponse extends Error {
  constructor() {
    super('daemon hook response exceeds the managed response limit');
    this.name = 'OversizedResponse';
  }
}

export async function postDaemonHook(
  endpoint: DaemonEndpoint,
  payload: Record<string, unknown>,
): Promise<HookOutcome> {
  const encoded = JSON.stringify(payload);
  if (Buffer.byteLength(encoded) > MAX_HOOK_PAYLOAD_BYTES) {
    return {
      kind: 'fault',
      origin: 'transport',
      detail: 'hook payload exceeds the managed payload limit',
    };
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HOOK_TIMEOUT_MS);
  try {
    const response = await fetch(`${endpoint.daemonAddress}/hooks/pi`, {
      method: 'POST',
      redirect: 'error',
      headers: {
        'content-type': 'application/json',
        [CLIENT_TOKEN_HEADER]: endpoint.credential,
      },
      body: encoded,
      signal: controller.signal,
    });
    return outcomeFromResponse(response.status, decodeBody(await readBounded(response)));
  } catch (error) {
    if (error instanceof OversizedResponse) {
      return { kind: 'fault', origin: 'response', detail: error.message };
    }
    if (error instanceof Error && error.name === 'AbortError') {
      return {
        kind: 'fault',
        origin: 'timeout',
        detail: `daemon did not answer within ${HOOK_TIMEOUT_MS}ms`,
      };
    }
    return {
      kind: 'fault',
      origin: 'transport',
      detail: `daemon hook request failed: ${errorDetail(error)}`,
    };
  } finally {
    clearTimeout(timer);
  }
}

async function readBounded(response: Response): Promise<Uint8Array> {
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  while (true) {
    const part = await reader.read();
    if (part.done) break;
    length += part.value.byteLength;
    if (length > MAX_HOOK_RESPONSE_BYTES) {
      await reader.cancel();
      throw new OversizedResponse();
    }
    chunks.push(part.value);
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function decodeBody(bytes: Uint8Array): unknown {
  try {
    return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch {
    return null;
  }
}
