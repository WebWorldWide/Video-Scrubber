// SPDX-License-Identifier: GPL-3.0-or-later
//! Supervises the local Python engine: first-run setup, spawn, ready detection and restarts.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::setup;

#[derive(Clone, Serialize)]
pub struct EngineState {
    state: &'static str, // installing | starting | ready | error
    url: Option<String>,
    token: String,
    step: String,
    pct: f64,
    log: Vec<String>,
    error: Option<String>,
}

struct Inner {
    pub_state: EngineState,
    full_log: Vec<String>,
    child: Option<Child>,
    restarts: u32,
    quitting: bool,
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<Mutex<Inner>>,
    app: AppHandle,
}

fn token() -> String {
    let mut b = [0u8; 24];
    getrandom::fill(&mut b).expect("OS random source");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

impl Engine {
    pub fn new(app: AppHandle) -> Self {
        let pub_state = EngineState { state: "starting", url: None, token: token(), step: String::new(), pct: 0.0, log: vec![], error: None };
        Engine { inner: Arc::new(Mutex::new(Inner { pub_state, full_log: vec![], child: None, restarts: 0, quitting: false })), app }
    }

    pub fn snapshot(&self) -> EngineState {
        let g = self.inner.lock().unwrap();
        let mut s = g.pub_state.clone();
        s.log = g.full_log.iter().rev().take(6).rev().cloned().collect();
        s
    }

    fn update(&self, f: impl FnOnce(&mut Inner)) {
        f(&mut self.inner.lock().unwrap());
        let _ = self.app.emit("engine", self.snapshot());
    }

    fn log(&self, line: String) {
        self.update(|g| {
            g.full_log.push(line.chars().take(300).collect());
            if g.full_log.len() > 200 {
                g.full_log.drain(..100);
            }
        });
    }

    fn engine_dir(&self) -> PathBuf {
        if cfg!(debug_assertions) {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("engine")
        } else {
            self.app.path().resource_dir().expect("resource dir").join("engine")
        }
    }

    fn venv_dir(&self) -> PathBuf {
        if cfg!(debug_assertions) {
            self.engine_dir().join(".venv")
        } else {
            self.app.path().app_local_data_dir().expect("data dir").join("engine-venv")
        }
    }

    /// Install if needed, then spawn. Runs on a background thread.
    pub fn start(&self) {
        let me = self.clone();
        std::thread::spawn(move || {
            let (dir, venv) = (me.engine_dir(), me.venv_dir());
            if !setup::is_ready(&dir, &venv) {
                me.update(|g| g.pub_state.state = "installing");
                let _ = std::fs::create_dir_all(venv.parent().unwrap_or(&venv));
                let bin = me.app.path().resource_dir().ok().map(|d| d.join("bin"));
                let step = |s: &str, p: f64| {
                    me.update(|g| {
                        g.pub_state.step = s.to_string();
                        g.pub_state.pct = p;
                    })
                };
                if let Err(e) = setup::setup(&dir, &venv, bin.as_deref(), &|l| me.log(l), &step) {
                    me.update(|g| {
                        g.pub_state.state = "error";
                        g.pub_state.error = Some(e);
                    });
                    return;
                }
            }
            me.update(|g| g.pub_state.state = "starting");
            me.spawn();
        });
    }

    fn spawn(&self) {
        let (dir, venv) = (self.engine_dir(), self.venv_dir());
        let tok = self.inner.lock().unwrap().pub_state.token.clone();
        let child = setup::command(setup::venv_python(&venv))
            .args(["-m", "vscrubber"])
            .current_dir(&dir)
            .env("VS_TOKEN", tok)
            .env("PYTHONPATH", &dir)
            .env("PYTHONUNBUFFERED", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                return self.update(|g| {
                    g.pub_state.state = "error";
                    g.pub_state.error = Some(format!("Couldn’t start the local engine: {e}"));
                })
            }
        };
        let (out, err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
        self.inner.lock().unwrap().child = Some(child);

        let me = self.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                if let Some(port) = line.strip_prefix("VSCRUBBER_READY port=") {
                    let url = format!("http://127.0.0.1:{}", port.trim());
                    me.update(|g| {
                        g.pub_state.state = "ready";
                        g.pub_state.url = Some(url);
                        g.pub_state.error = None;
                        g.restarts = 0;
                    });
                }
            }
        });
        let me = self.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                let mut g = me.inner.lock().unwrap();
                g.full_log.push(line.chars().take(300).collect());
                if g.full_log.len() > 200 {
                    g.full_log.drain(..100);
                }
            }
        });
        let me = self.clone();
        std::thread::spawn(move || me.watch());
    }

    /// Restart a crashed engine up to three times, then surface the error.
    fn watch(&self) {
        let status = loop {
            std::thread::sleep(Duration::from_millis(500));
            let mut g = self.inner.lock().unwrap();
            match g.child.as_mut().map(|c| c.try_wait()) {
                Some(Ok(Some(st))) => {
                    g.child = None;
                    break Some(st);
                }
                Some(Ok(None)) => continue,
                _ => break None,
            }
        };
        let (quitting, restarts) = {
            let g = self.inner.lock().unwrap();
            (g.quitting, g.restarts)
        };
        if quitting {
            return;
        }
        if restarts < 3 {
            self.update(|g| {
                g.restarts += 1;
                g.pub_state.state = "starting";
                g.pub_state.url = None;
            });
            std::thread::sleep(Duration::from_secs(1 + restarts as u64));
            self.spawn();
        } else {
            let tail = self.inner.lock().unwrap().full_log.iter().rev().take(3).rev().cloned().collect::<Vec<_>>().join(" ");
            let code = status.map(|s| s.to_string()).unwrap_or_else(|| "unknown".into());
            self.update(|g| {
                g.pub_state.state = "error";
                g.pub_state.url = None;
                g.pub_state.error = Some(format!("The local engine stopped ({code}). {tail}"));
            });
        }
    }

    pub fn restart(&self) {
        let running = {
            let mut g = self.inner.lock().unwrap();
            g.restarts = 0;
            g.pub_state.error = None;
            match g.child.as_mut() {
                Some(c) => {
                    let _ = c.kill();
                    true
                }
                None => false,
            }
        };
        // A killed engine is respawned by `watch`; otherwise start from scratch (setup may have failed).
        if !running {
            self.start();
        }
    }

    pub fn shutdown(&self) {
        let mut g = self.inner.lock().unwrap();
        g.quitting = true;
        if let Some(c) = g.child.as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}
