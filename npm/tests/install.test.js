'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { safeReleaseUrl, sha256, get, readArchiveExecutable } = require('../install.js');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { execFileSync } = require('node:child_process');

test('allows canonical GitHub release assets and relative redirects', () => {
  assert.equal(
    safeReleaseUrl('https://github.com/lacausecrypto/mcp-sophon/releases/download/v0.5.4/file.tar.gz'),
    'https://github.com/lacausecrypto/mcp-sophon/releases/download/v0.5.4/file.tar.gz'
  );
  assert.equal(
    safeReleaseUrl('/asset', 'https://github.com/release'),
    'https://github.com/asset'
  );
  assert.equal(
    safeReleaseUrl('https://release-assets.githubusercontent.com/asset'),
    'https://release-assets.githubusercontent.com/asset'
  );
});

test('rejects HTTP, lookalike domains, custom ports and credentials', () => {
  for (const url of [
    'http://github.com/asset',
    'https://github.com.attacker.invalid/asset',
    'https://evilgithubusercontent.com/asset',
    'https://github.com:8443/asset',
    'https://username:secret@github.com/asset',
    'file:///tmp/asset'
  ]) {
    assert.throws(() => safeReleaseUrl(url), /untrusted release download URL/);
  }
});

test('rejects untrusted redirect targets without downloading them', async () => {
  await assert.rejects(
    get('https://example.com/release.tar.gz'),
    /untrusted release download URL/
  );
});

test('SHA-256 helper agrees with the known digest of empty content', () => {
  assert.equal(
    sha256(Buffer.alloc(0)),
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
  );
});

function archiveFixture(t, { invalidSignature = false, symlink = false } = {}) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'sophon-archive-test-'));
  t.after(() => fs.rmSync(tmp, { recursive: true, force: true }));
  const windows = process.platform === 'win32';
  const target = windows
    ? 'x86_64-pc-windows-msvc'
    : process.platform === 'darwin'
      ? (process.arch === 'arm64' ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin')
      : (process.arch === 'arm64' ? 'aarch64-unknown-linux-gnu' : 'x86_64-unknown-linux-gnu');
  const folder = `sophon-${target}`;
  const folderPath = path.join(tmp, folder);
  fs.mkdirSync(folderPath);
  const executable = path.join(folderPath, windows ? 'sophon.exe' : 'sophon');
  const binary = Buffer.alloc(8192);
  if (windows) binary.write('MZ', 0, 'ascii');
  else if (process.platform === 'darwin') Buffer.from('cffaedfe', 'hex').copy(binary);
  else Buffer.from('7f454c46', 'hex').copy(binary);
  if (invalidSignature) binary[0] = 0;
  if (symlink) fs.symlinkSync(path.join(tmp, 'missing-outside'), executable);
  else fs.writeFileSync(executable, binary);
  // A malicious, unrelated entry must never be extracted to the filesystem.
  fs.writeFileSync(path.join(tmp, 'unrelated.txt'), 'DO NOT EXTRACT');
  const archiveName = `${folder}.${windows ? 'zip' : 'tar.gz'}`;
  const archivePath = path.join(tmp, archiveName);
  if (windows) {
    execFileSync('tar.exe', ['-a', '-cf', archivePath, '-C', tmp, folder, 'unrelated.txt']);
  } else {
    execFileSync('tar', ['-czf', archivePath, '-C', tmp, folder, 'unrelated.txt']);
  }
  fs.rmSync(folderPath, { recursive: true, force: true });
  fs.rmSync(path.join(tmp, 'unrelated.txt'));
  return { tmp, archivePath, archiveName, binary };
}

test('extracts only the expected release executable, without creating archive paths', (t) => {
  const { tmp, archivePath, archiveName, binary } = archiveFixture(t);
  const extracted = readArchiveExecutable(archivePath, archiveName);
  assert.deepEqual(extracted, binary);
  assert.equal(fs.existsSync(path.join(tmp, 'unrelated.txt')), false);
});

test('rejects a release member with an invalid executable header', (t) => {
  const { archivePath, archiveName } = archiveFixture(t, { invalidSignature: true });
  assert.throws(
    () => readArchiveExecutable(archivePath, archiveName),
    /not a valid executable/
  );
});

test('rejects an archived symlink instead of following its target', {
  skip: process.platform === 'win32',
}, (t) => {
  const { archivePath, archiveName } = archiveFixture(t, { symlink: true });
  assert.throws(
    () => readArchiveExecutable(archivePath, archiveName)
  );
});
