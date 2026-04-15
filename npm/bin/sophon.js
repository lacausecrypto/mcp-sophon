#!/usr/bin/env node
/**
 * Thin shim that forwards argv + stdio to the native `sophon` binary
 * downloaded by install.js into ./bin/sophon (or sophon.exe on Windows).
 *
 * Why a JS shim instead of a direct `bin` pointing at the binary?
 *   npm on Windows will not shell out to a raw .exe via the "bin" field
 *   consistently across shells. A Node shim works everywhere.
 */
'use strict';

const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const binName = process.platform === 'win32' ? 'sophon.exe' : 'sophon';
const binary = path.join(__dirname, binName);

if (!fs.existsSync(binary)) {
  console.error(`[sophon] native binary not found at ${binary}`);
  console.error('[sophon] re-run `npm install` or set SOPHON_SKIP_DOWNLOAD=0');
  process.exit(1);
}

const child = spawn(binary, process.argv.slice(2), { stdio: 'inherit' });
child.on('close', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 0);
  }
});
child.on('error', (e) => {
  console.error(`[sophon] failed to spawn ${binary}: ${e.message}`);
  process.exit(1);
});
