// SPDX-License-Identifier: GPL-3.0-or-later
//! Builds the Python environment for the local engine, choosing the PyTorch build
//! that accelerates on this machine (Metal, CUDA, ROCm, Intel XPU, DirectML or CPU).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const TORCH_INDEX: &str = "https://download.pytorch.org/whl/";

pub struct Accel {
    pub id: &'static str,
    pub label: String,
    torch_index: Option<String>,
    extras: Vec<&'static str>,
}

/// A `Command` that never flashes a console window on Windows.
pub fn command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    #[allow(unused_mut)]
    let mut c = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    c
}

fn output(program: &str, args: &[&str]) -> Option<String> {
    let out = command(program).args(args).stderr(Stdio::null()).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn detect_accel() -> Accel {
    let idx = |s: &str| Some(format!("{TORCH_INDEX}{s}"));
    if cfg!(target_os = "macos") {
        return if cfg!(target_arch = "aarch64") {
            Accel { id: "apple", label: "Apple Silicon (Metal)".into(), torch_index: None, extras: vec!["apple"] }
        } else {
            Accel { id: "cpu", label: "Intel Mac (CPU)".into(), torch_index: None, extras: vec![] }
        };
    }
    if let Some(gpu) = output("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]).filter(|s| !s.trim().is_empty()) {
        let name = gpu.lines().next().unwrap_or("NVIDIA GPU").trim().to_string();
        return Accel { id: "cuda", label: format!("{name} (CUDA)"), torch_index: idx("cu128"), extras: vec!["cuda"] };
    }
    if cfg!(target_os = "linux") {
        let rocm = output("rocminfo", &[]).is_some_and(|s| s.contains("gfx")) || Path::new("/opt/rocm").exists();
        if rocm {
            return Accel { id: "rocm", label: "AMD GPU (ROCm)".into(), torch_index: idx("rocm6.4"), extras: vec![] };
        }
        let pci = output("lspci", &[]).unwrap_or_default();
        if pci.lines().any(|l| (l.contains("VGA") || l.contains("Display")) && l.contains("Intel") && (l.contains("Arc") || l.contains("Xe"))) {
            return Accel { id: "xpu", label: "Intel GPU (XPU)".into(), torch_index: idx("xpu"), extras: vec![] };
        }
        return Accel { id: "cpu", label: "CPU".into(), torch_index: idx("cpu"), extras: vec![] };
    }
    if cfg!(windows) {
        let gpus = output("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"]).unwrap_or_default();
        if gpus.contains("Arc") && gpus.contains("Intel") {
            return Accel { id: "xpu", label: "Intel Arc (XPU)".into(), torch_index: idx("xpu"), extras: vec![] };
        }
        if ["AMD", "Radeon", "Intel"].iter().any(|k| gpus.contains(k)) {
            let name = gpus.lines().next().unwrap_or("GPU").trim().to_string();
            return Accel { id: "directml", label: format!("{name} (DirectML)"), torch_index: None, extras: vec!["directml"] };
        }
    }
    Accel { id: "cpu", label: "CPU".into(), torch_index: None, extras: vec![] }
}

fn find_uv(resource_bin: Option<&Path>) -> Option<PathBuf> {
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    let mut cands: Vec<PathBuf> = vec![];
    if let Ok(p) = std::env::var("VS_UV") {
        cands.push(p.into());
    }
    if let Some(b) = resource_bin {
        cands.push(b.join(exe));
    }
    if let Some(home) = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
        cands.push(PathBuf::from(&home).join(".local").join("bin").join(exe));
        cands.push(PathBuf::from(&home).join(".cargo").join("bin").join(exe));
    }
    if let Some(p) = cands.into_iter().find(|p| p.exists()) {
        return Some(p);
    }
    output(exe, &["--version"]).map(|_| PathBuf::from(exe))
}

fn find_python() -> Option<(String, Vec<String>)> {
    let cands: &[(&str, &[&str])] = if cfg!(windows) {
        &[("py", &["-3.12"]), ("py", &["-3.11"]), ("py", &["-3.13"]), ("python", &[])]
    } else {
        &[("python3.12", &[]), ("python3.11", &[]), ("python3.13", &[]), ("python3.10", &[]), ("python3", &[])]
    };
    for (cmd, pre) in cands {
        let mut args: Vec<&str> = pre.to_vec();
        args.extend(["-c", "import sys;print((3,10)<=sys.version_info[:2]<(3,14))"]);
        if output(cmd, &args).is_some_and(|s| s.trim() == "True") {
            return Some((cmd.to_string(), pre.iter().map(|s| s.to_string()).collect()));
        }
    }
    None
}

pub fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

/// FNV-1a of pyproject.toml, so a changed dependency list triggers a re-install.
fn spec_hash(engine_dir: &Path) -> String {
    let bytes = std::fs::read(engine_dir.join("pyproject.toml")).unwrap_or_default();
    let h = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100_0000_01b3));
    format!("{h:016x}")
}

pub fn is_ready(engine_dir: &Path, venv: &Path) -> bool {
    let marker = std::fs::read_to_string(venv.join(".vs-setup.json")).unwrap_or_default();
    marker.contains(&format!("\"spec\":\"{}\"", spec_hash(engine_dir))) && venv_python(venv).exists()
}

fn run(program: &Path, args: &[String], log: &(dyn Fn(String) + Sync)) -> Result<(), String> {
    let name = program.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    log(format!("$ {name} {}", args.join(" ")));
    let mut child = command(program)
        .args(args)
        .env("UV_NO_PROGRESS", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Couldn’t run {name}: {e}"))?;
    let err = child.stderr.take().unwrap();
    let errs = std::thread::scope(|s| {
        let h = s.spawn(|| BufReader::new(err).lines().map_while(Result::ok).inspect(|l| log(l.clone())).collect::<Vec<_>>());
        BufReader::new(child.stdout.take().unwrap()).lines().map_while(Result::ok).for_each(|l| log(l));
        h.join().unwrap_or_default()
    });
    let status = child.wait().map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{name} exited with {status}: {}", errs.last().cloned().unwrap_or_default()))
    }
}

/// pip can't read [project] deps from a pyproject without building it, so pull them out.
fn read_deps(pyproject: &Path, extras: &[&str]) -> Vec<String> {
    let src = std::fs::read_to_string(pyproject).unwrap_or_default();
    let block = |name: &str| -> Vec<String> {
        let Some(start) = src.find(&format!("{name} = [")) else { return vec![] };
        let body = &src[start..];
        let end = body.find(']').unwrap_or(body.len());
        body[..end].split('"').skip(1).step_by(2).map(String::from).collect()
    };
    let mut deps = block("dependencies");
    for e in extras {
        deps.extend(block(e));
    }
    deps
}

pub fn setup(engine_dir: &Path, venv: &Path, resource_bin: Option<&Path>, log: &(dyn Fn(String) + Sync), step: &dyn Fn(&str, f64)) -> Result<Accel, String> {
    let accel = detect_accel();
    log(format!("Accelerator: {}", accel.label));
    let py = venv_python(venv);
    let pyproject = engine_dir.join("pyproject.toml");
    let s = |p: &Path| p.to_string_lossy().into_owned();
    let v = |xs: &[&str]| xs.iter().map(|x| x.to_string()).collect::<Vec<_>>();

    step("Creating Python environment", 0.05);
    if let Some(uv) = find_uv(resource_bin) {
        run(&uv, &[v(&["venv", "--allow-existing", "--python", "3.12"]), vec![s(venv)]].concat(), log)?;
        if let Some(idx) = &accel.torch_index {
            step(&format!("Installing PyTorch for {}", accel.label), 0.2);
            // torchvision must come from the same index so its ops match this torch build.
            run(&uv, &[v(&["pip", "install", "--python"]), vec![s(&py)], v(&["torch", "torchvision", "--index-url"]), vec![idx.clone()]].concat(), log)?;
        }
        step("Installing speech, vision and search runtimes", 0.5);
        let mut args = [v(&["pip", "install", "--python"]), vec![s(&py), "-r".into(), s(&pyproject)]].concat();
        for e in &accel.extras {
            args.extend(["--extra".into(), e.to_string()]);
        }
        run(&uv, &args, log)?;
    } else {
        let (cmd, pre) = find_python().ok_or("Python 3.10–3.13 or uv is required to set up the local engine. Install uv from https://docs.astral.sh/uv/ and reopen Video Scrubber.")?;
        let cmd = PathBuf::from(cmd);
        run(&cmd, &[pre, v(&["-m", "venv"]), vec![s(venv)]].concat(), log)?;
        run(&py, &v(&["-m", "pip", "install", "--upgrade", "pip"]), log)?;
        if let Some(idx) = &accel.torch_index {
            step(&format!("Installing PyTorch for {}", accel.label), 0.2);
            run(&py, &[v(&["-m", "pip", "install", "torch", "torchvision", "--index-url"]), vec![idx.clone()]].concat(), log)?;
        }
        step("Installing speech, vision and search runtimes", 0.5);
        run(&py, &[v(&["-m", "pip", "install"]), read_deps(&pyproject, &accel.extras)].concat(), log)?;
    }
    let marker = format!("{{\"spec\":\"{}\",\"accel\":\"{}\"}}", spec_hash(engine_dir), accel.id);
    std::fs::write(venv.join(".vs-setup.json"), marker).map_err(|e| e.to_string())?;
    step("Ready", 1.0);
    Ok(accel)
}
