'use strict';

// Runs the *current PR installer* against assets that are already published
// in a real GitHub Release. Does not publish anything or modify npm.
// CI pins an existing version; the local package.json remains untouched.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const release = process.env.SOPHON_TEST_RELEASE_VERSION;
if (!release || !/^\d+\.\d+\.\d+$/.test(release)) {
  throw new Error('SOPHON_TEST_RELEASE_VERSION must specify a published semver, e.g. 0.7.0');
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sophon-release-e2e-'));
try {
  fs.copyFileSync(path.join(__dirname, '..', 'install.js'), path.join(root, 'install.js'));
  fs.writeFileSync(
    path.join(root, 'package.json'),
    JSON.stringify({ name: 'sophon-release-e2e', version: release }, null, 2)
  );
  const install = spawnSync(process.execPath, ['install.js'], {
    cwd: root,
    env: { ...process.env, SOPHON_SKIP_DOWNLOAD: '0' },
    encoding: 'utf8',
    timeout: 90000,
    maxBuffer: 1024 * 1024,
    windowsHide: true,
  });
  assert.equal(
    install.status, 0,
    `live release installer exited ${install.status}; signal=${install.signal}\n${install.stdout}\n${install.stderr}`
  );
  assert.match(install.stdout, /sha256 verified:/);
  const executable = path.join(root, 'bin', process.platform === 'win32' ? 'sophon.exe' : 'sophon');
  assert.ok(fs.statSync(executable).size > 100000, 'installed binary is unexpectedly small');

  const rpc = [
    { jsonrpc: '2.0', id: 1, method: 'initialize', params: {
      protocolVersion: '2025-06-18',
      capabilities: {},
      clientInfo: { name: 'release-e2e', version: '1' },
    } },
    { jsonrpc: '2.0', id: 2, method: 'tools/call', params: {
      name: 'count_tokens', arguments: { text: 'hello world' },
    } },
  ];
  const smoke = spawnSync(executable, ['serve'], {
    input: rpc.map(m => JSON.stringify(m)).join('\n') + '\n',
    encoding: 'utf8',
    timeout: 20000,
    maxBuffer: 2 * 1024 * 1024,
    windowsHide: true,
  });
  assert.equal(smoke.status, 0, `published binary failed: ${smoke.stderr}\n${smoke.stdout}`);
  const responses = smoke.stdout.split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
  assert.ok(responses.some(r => r.id === 1 && r.result), 'initialize response missing');
  assert.ok(responses.some(r => r.id === 2 && r.result), 'count_tokens result missing');
  console.log(`Verified release v${release}: checksum, binary install and live MCP handshake (${process.platform}-${process.arch})`);
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
