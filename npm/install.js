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
const zlib = require('zlib');
const { execSync } = require('child_process');

const REPO = process.env.SOPHON_REPO || 'lacausecrypto/mcp-sophon';
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

function get(url, redirects = 5) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { 'User-Agent': `sophon-npm/${pkg.version}` } }, (res) => {
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location && redirects > 0) {
        res.resume();
        return get(res.headers.location, redirects - 1).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode} on ${url}`));
      }
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    });
    req.on('error', reject);
  });
}

function sha256(buf) {
  return crypto.createHash('sha256').update(buf).digest('hex');
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
    const shaBuf = await get(sha256Url);
    expectedSha = shaBuf.toString('utf8').trim().split(/\s+/)[0].toLowerCase();
  } catch (e) {
    console.warn(`[sophon] WARNING: no .sha256 file at ${sha256Url} — skipping verification`);
  }

  if (expectedSha) {
    const actual = sha256(archive);
    if (actual !== expectedSha) {
      fail(`checksum mismatch: expected ${expectedSha}, got ${actual}`);
    }
    console.log(`[sophon] sha256 verified: ${expectedSha}`);
  }

  // Extract the archive. For tar.gz we prefer the system tar to avoid adding a
  // dependency; for .zip on Windows we shell out to PowerShell's Expand-Archive.
  const tmpDir = path.join(__dirname, '.tmp-extract');
  fs.rmSync(tmpDir, { recursive: true, force: true });
  fs.mkdirSync(tmpDir, { recursive: true });
  const archivePath = path.join(tmpDir, archiveName);
  fs.writeFileSync(archivePath, archive);

  try {
    if (archiveName.endsWith('.tar.gz')) {
      execSync(`tar -xzf "${archivePath}" -C "${tmpDir}"`, { stdio: 'inherit' });
    } else if (archiveName.endsWith('.zip')) {
      // Windows ships PowerShell; Unix environments that hit this branch already have unzip.
      if (isWindows) {
        execSync(`powershell -NoProfile -Command "Expand-Archive -Force -Path '${archivePath}' -DestinationPath '${tmpDir}'"`, { stdio: 'inherit' });
      } else {
        execSync(`unzip -o "${archivePath}" -d "${tmpDir}"`, { stdio: 'inherit' });
      }
    } else {
      fail(`unknown archive format: ${archiveName}`);
    }
  } catch (e) {
    fail(`extraction failed: ${e.message}`);
  }

  // The archive contains a folder like sophon-<target>/ with the binary inside.
  const subdirs = fs.readdirSync(tmpDir).filter((n) => fs.statSync(path.join(tmpDir, n)).isDirectory());
  let found = null;
  for (const d of subdirs) {
    const candidate = path.join(tmpDir, d, isWindows ? 'sophon.exe' : 'sophon');
    if (fs.existsSync(candidate)) {
      found = candidate;
      break;
    }
  }
  if (!found) {
    fail('binary not found inside extracted archive');
  }
  fs.copyFileSync(found, finalBinary);
  if (!isWindows) {
    fs.chmodSync(finalBinary, 0o755);
  }
  fs.rmSync(tmpDir, { recursive: true, force: true });

  console.log(`[sophon] installed to ${finalBinary}`);
}

main().catch((e) => fail(e.stack || e.message));
