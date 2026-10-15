'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { safeReleaseUrl, sha256, get } = require('../install.js');

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
