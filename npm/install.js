#!/usr/bin/env node
/**
 * Sophon npm wrapper — postinstall binary fetcher.
 *
 * Downloads the prebuilt `sophon` binary matching the current platform/arch
 * from the GitHub Releases page for the version in package.json, verifies
 * the SHA-256, and drops it next to bin/sophon.js.
 *
 * No network dependency other than `https.get` from Node's stdlib.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const https = require('https');
const { execFileSync } = require('child_process');
const os = require('os');

const REPO = 'lacausecrypto/mcp-sophon';
const pkg = require('./package.json');
const VERSION = `v${pkg.version}`;

const TARGETS = {
  'darwin-arm64': 'sophon-aarch64-apple-darwin.tar.gz',
  'darwin-x64':   'sophon-x86_64-apple-darwin.tar.gz',
  'linux-arm64':  'sophon-aarch64-unknown-linux-gnu.tar.gz',
  'linux-x64':    'sophon-x86_64-unknown-linux-gnu.tar.gz',
  'win32-x64':    'sophon-x86_64-pc-windows-msvc.zip',
};

function key() {
  return `${process.platform}-${process.arch}`;
}

function fail(msg) {
  console.error(`[sophon] ${msg}`);
  console.error(`[sophon] You can build from source instead:`);
  console.error(`[sophon]   git clone https://github.com/${REPO}`);
  console.error(`[sophon]   cd mcp-sophon/sophon && cargo build --release -p mcp-integration`);
  process.exit(1);
}

// GitHub Releases download URLs may redirect to GitHub's asset CDN.
// Reject untrusted destinations before making a network request.
function safeReleaseUrl(url, relativeTo) {
  const parsed = new URL(url, relativeTo);
  const host = parsed.hostname.toLowerCase();
  if (parsed.protocol !== 'https:' || parsed.username || parsed.password ||
      parsed.port || (host !== 'github.com' && !host.endsWith('.githubusercontent.com'))) {
    throw new Error(`untrusted release download URL: ${parsed.origin}`);
  }
  return parsed.href;
}

function get(url, redirects = 5, maxBytes = 128 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let target;
    try {
      target = safeReleaseUrl(url);
    } catch (e) {
      return reject(e);
    }
    const req = https.get(target, { headers: { 'User-Agent': `sophon-npm/${pkg.version}` } }, (res) => {
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location && redirects > 0) {
        res.resume();
        let redirect;
        try {
          redirect = safeReleaseUrl(res.headers.location, target);
        } catch (e) {
          return reject(e);
        }
        return get(redirect, redirects - 1, maxBytes).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`HTTP ${res.statusCode} on ${target}`));
      }
      const chunks = [];
      let size = 0;
      res.on('data', (c) => {
        size += c.length;
        if (size > maxBytes) {
          res.destroy(new Error(`release download exceeds ${maxBytes} bytes`));
          return;
        }
        chunks.push(c);
      });
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
      res.on('aborted', () => reject(new Error('release download interrupted')));
    });
    req.setTimeout(30000, () => req.destroy(new Error('release download timed out')));
    req.on('error', reject);
  });
}

function sha256(buf) {
  return crypto.createHash('sha256').update(buf).digest('hex');
}

// Read *only* the expected binary entry to stdout. Never extract arbitrary
// archive members into the filesystem: tarballs and ZIPs can contain traversal,
// symlinks or hardlinks with effects beyond the temporary directory.
function readArchiveExecutable(archivePath, archiveName, platformKey = key()) {
  const target = archiveName.replace(/^sophon-/, '').replace(/\.(tar\.gz|zip)$/, '');
  const executable = platformKey.startsWith('win32-') ? 'sophon.exe' : 'sophon';
  const member = `sophon-${target}/${executable}`;
  const tool = platformKey.startsWith('win32-') ? 'tar.exe' : 'tar';
  // Windows ships libarchive's tar.exe, which supports both ZIP and tar files.
  const binary = execFileSync(tool, ['-xOf', archivePath, member], {
    maxBuffer: 64 * 1024 * 1024,
    timeout: 30000,
    windowsHide: true,
  });
  const header = binary.subarray(0, 4);
  const valid = platformKey.startsWith('linux-')
    ? header.equals(Buffer.from([0x7f, 0x45, 0x4c, 0x46])) // ELF
    : platformKey.startsWith('darwin-')
      ? ['cffaedfe', 'feedfacf', 'cafebabe', 'bebafeca', 'cafebabf', 'bfbafeca'].includes(header.toString('hex')) // Mach-O
      : binary.subarray(0, 2).toString('ascii') === 'MZ'; // PE
  if (binary.length < 4096 || !valid) {
    throw new Error(`archive entry ${member} is not a valid executable for ${platformKey}`);
  }
  return binary;
}

async function main() {
  // Skip the whole thing if SOPHON_SKIP_DOWNLOAD is set — useful in CI / Docker.
  if (process.env.SOPHON_SKIP_DOWNLOAD === '1') {
    console.log('[sophon] SOPHON_SKIP_DOWNLOAD=1, skipping.');
    return;
  }

  const k = key();
  const archiveName = TARGETS[k];
  if (!archiveName) {
    fail(`Unsupported platform ${k}. Supported: ${Object.keys(TARGETS).join(', ')}`);
  }

  const binDir = path.join(__dirname, 'bin');
  fs.mkdirSync(binDir, { recursive: true });
  const isWindows = process.platform === 'win32';
  const finalBinary = path.join(binDir, isWindows ? 'sophon.exe' : 'sophon');

  // If the binary was dropped manually or by a previous run, leave it alone.
  if (fs.existsSync(finalBinary)) {
    console.log(`[sophon] binary already present at ${finalBinary}`);
    return;
  }

  const base = `https://github.com/${REPO}/releases/download/${VERSION}`;
  const archiveUrl = `${base}/${archiveName}`;
  const sha256Url  = `${archiveUrl}.sha256`;

  console.log(`[sophon] downloading ${archiveUrl}`);
  let archive, expectedSha;
  try {
    archive = await get(archiveUrl);
  } catch (e) {
    fail(`failed to download ${archiveUrl}: ${e.message}`);
  }
  try {
    const shaBuf = await get(sha256Url, 5, 4096);
    expectedSha = shaBuf.toString('utf8').trim().split(/\s+/)[0].toLowerCase();
  } catch (e) {
    fail(`required SHA-256 checksum unavailable at ${sha256Url}: ${e.message}`);
  }

  if (!/^[0-9a-f]{64}$/.test(expectedSha)) fail('invalid SHA-256 checksum format');
  const actual = sha256(archive);
  if (actual !== expectedSha) fail(`checksum mismatch: expected ${expectedSha}, got ${actual}`);
  console.log(`[sophon] sha256 verified: ${expectedSha}`);

  // Stage the verified archive in an isolated temporary directory, and
  // decode exactly its expected executable (never arbitrary archive entries).
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sophon-install-'));
  try {
    const archivePath = path.join(tmpDir, archiveName);
    fs.writeFileSync(archivePath, archive, { mode: 0o600 });
    const binary = readArchiveExecutable(archivePath, archiveName, k);
    const stagedBinary = path.join(tmpDir, isWindows ? 'sophon.exe' : 'sophon');
    fs.writeFileSync(stagedBinary, binary, { mode: 0o755 });
    if (!isWindows) fs.chmodSync(stagedBinary, 0o755);
    // Never overwrite another process's already-present binary.
    fs.copyFileSync(stagedBinary, finalBinary, fs.constants.COPYFILE_EXCL);
    if (!isWindows) fs.chmodSync(finalBinary, 0o755);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }

  console.log(`[sophon] installed to ${finalBinary}`);
}

if (require.main === module) {
  main().catch((e) => fail(e.stack || e.message));
}

module.exports = { safeReleaseUrl, sha256, get, readArchiveExecutable };
