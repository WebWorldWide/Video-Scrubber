// Builds the Python environment for the local engine, choosing the PyTorch build
// that accelerates on this machine (Metal, CUDA, ROCm, Intel XPU, DirectML or CPU).
const { spawn, spawnSync } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const IS_WIN = process.platform === 'win32';
const TORCH_INDEX = 'https://download.pytorch.org/whl/';

function ok(cmd, args) {
  try {
    return spawnSync(cmd, args, { encoding: 'utf8', timeout: 15000, windowsHide: true });
  } catch {
    return null;
  }
}

function detectAccel() {
  if (process.platform === 'darwin') {
    return os.arch() === 'arm64' ? { id: 'apple', label: 'Apple Silicon (Metal)', extras: ['apple'] } : { id: 'cpu', label: 'Intel Mac (CPU)', extras: [] };
  }
  const smi = ok('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader']);
  if (smi && smi.status === 0 && smi.stdout.trim()) {
    return { id: 'cuda', label: smi.stdout.trim().split('\n')[0] + ' (CUDA)', torchIndex: TORCH_INDEX + 'cu128', extras: ['cuda'] };
  }
  if (process.platform === 'linux') {
    const rocm = ok('rocminfo', []);
    if ((rocm && rocm.status === 0 && /gfx\d+/.test(rocm.stdout)) || fs.existsSync('/opt/rocm')) {
      return { id: 'rocm', label: 'AMD GPU (ROCm)', torchIndex: TORCH_INDEX + 'rocm6.4', extras: [] };
    }
    const pci = ok('lspci', []);
    if (pci && /VGA.*Intel.*(Arc|Xe)|Display.*Intel.*(Arc|Xe)/i.test(pci.stdout || '')) {
      return { id: 'xpu', label: 'Intel GPU (XPU)', torchIndex: TORCH_INDEX + 'xpu', extras: [] };
    }
    return { id: 'cpu', label: 'CPU', torchIndex: TORCH_INDEX + 'cpu', extras: [] };
  }
  if (IS_WIN) {
    const ps = ok('powershell', ['-NoProfile', '-Command', '(Get-CimInstance Win32_VideoController).Name']);
    const gpus = (ps && ps.stdout) || '';
    if (/Intel.*Arc/i.test(gpus)) return { id: 'xpu', label: 'Intel Arc (XPU)', torchIndex: TORCH_INDEX + 'xpu', extras: [] };
    if (/AMD|Radeon|Intel/i.test(gpus)) return { id: 'directml', label: gpus.split('\n')[0].trim() + ' (DirectML)', extras: ['directml'] };
  }
  return { id: 'cpu', label: 'CPU', extras: [] };
}

function findUv(resourcesBin) {
  const exe = IS_WIN ? 'uv.exe' : 'uv';
  const cands = [process.env.VS_UV, resourcesBin && path.join(resourcesBin, exe)].filter(Boolean);
  for (const c of cands) if (fs.existsSync(c)) return c;
  const r = ok(exe, ['--version']);
  return r && r.status === 0 ? exe : null;
}

function findPython() {
  const cands = IS_WIN
    ? [['py', ['-3.12']], ['py', ['-3.11']], ['py', ['-3.13']], ['python', []]]
    : [['python3.12', []], ['python3.11', []], ['python3.13', []], ['python3.10', []], ['python3', []]];
  for (const [cmd, pre] of cands) {
    const r = ok(cmd, [...pre, '-c', 'import sys;print(sys.version_info[:2] >= (3,10) and sys.version_info[:2] < (3,14))']);
    if (r && r.status === 0 && r.stdout.trim() === 'True') return [cmd, pre];
  }
  return null;
}

function venvPython(venvDir) {
  return IS_WIN ? path.join(venvDir, 'Scripts', 'python.exe') : path.join(venvDir, 'bin', 'python');
}

function specHash(engineDir) {
  return crypto.createHash('sha1').update(fs.readFileSync(path.join(engineDir, 'pyproject.toml'))).digest('hex').slice(0, 12);
}

function isReady(engineDir, venvDir) {
  try {
    const m = JSON.parse(fs.readFileSync(path.join(venvDir, '.vs-setup.json'), 'utf8'));
    return m.spec === specHash(engineDir) && fs.existsSync(venvPython(venvDir));
  } catch {
    return false;
  }
}

function run(cmd, args, log) {
  return new Promise((resolve, reject) => {
    log(`$ ${path.basename(cmd)} ${args.join(' ')}`);
    const p = spawn(cmd, args, { windowsHide: true, env: { ...process.env, UV_NO_PROGRESS: '1' } });
    const onData = d => String(d).split(/\r?\n/).filter(Boolean).forEach(l => log(l));
    p.stdout.on('data', onData);
    p.stderr.on('data', onData);
    p.on('error', reject);
    p.on('close', code => (code === 0 ? resolve() : reject(new Error(`${path.basename(cmd)} exited with ${code}`))));
  });
}

async function setupEngine({ engineDir, venvDir, resourcesBin, log = console.log, step = () => {} }) {
  const accel = detectAccel();
  log(`Accelerator: ${accel.label}`);
  const uv = findUv(resourcesBin);
  const py = venvPython(venvDir);
  const pyproject = path.join(engineDir, 'pyproject.toml');
  const extras = accel.extras.flatMap(e => ['--extra', e]);

  step('Creating Python environment', 0.05);
  if (uv) {
    await run(uv, ['venv', '--allow-existing', '--python', '3.12', venvDir], log);
    if (accel.torchIndex) {
      step('Installing PyTorch for ' + accel.label, 0.2);
      await run(uv, ['pip', 'install', '--python', py, 'torch', '--index-url', accel.torchIndex], log);
    }
    step('Installing speech, vision and search models runtime', 0.5);
    await run(uv, ['pip', 'install', '--python', py, '-r', pyproject, ...extras], log);
  } else {
    const found = findPython();
    if (!found) throw new Error('Python 3.10–3.13 or uv is required to set up the local engine.');
    await run(found[0], [...found[1], '-m', 'venv', venvDir], log);
    await run(py, ['-m', 'pip', 'install', '--upgrade', 'pip'], log);
    if (accel.torchIndex) {
      step('Installing PyTorch for ' + accel.label, 0.2);
      await run(py, ['-m', 'pip', 'install', 'torch', '--index-url', accel.torchIndex], log);
    }
    step('Installing speech, vision and search models runtime', 0.5);
    const deps = readDeps(pyproject, accel.extras);
    await run(py, ['-m', 'pip', 'install', ...deps], log);
  }
  fs.writeFileSync(path.join(venvDir, '.vs-setup.json'), JSON.stringify({ spec: specHash(engineDir), accel: accel.id, at: Date.now() }));
  step('Ready', 1);
  return { python: py, accel };
}

// pip can't read [project] deps from a pyproject without building it, so pull them out.
function readDeps(pyproject, extras) {
  const src = fs.readFileSync(pyproject, 'utf8');
  const block = name => {
    const m = src.match(new RegExp(name + '\\s*=\\s*\\[([\\s\\S]*?)\\]'));
    return m ? [...m[1].matchAll(/"([^"]+)"/g)].map(x => x[1]) : [];
  };
  return [...block('dependencies'), ...extras.flatMap(e => block(e))];
}

module.exports = { setupEngine, isReady, venvPython, detectAccel };

if (require.main === module) {
  const root = path.resolve(__dirname, '..');
  setupEngine({ engineDir: path.join(root, 'engine'), venvDir: path.join(root, 'engine', '.venv') })
    .then(r => console.log(`\nEngine ready (${r.accel.label}): ${r.python}`))
    .catch(e => {
      console.error(e.message);
      process.exit(1);
    });
}
