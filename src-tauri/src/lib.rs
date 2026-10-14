mod engine;
mod setup;

use std::path::Path;
use std::sync::Mutex;

use engine::{Engine, EngineState};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State, WebviewUrl, WebviewWindowBuilder};

const MEDIA_EXT: &[&str] = &[
    "mp4", "mov", "m4v", "mkv", "webm", "avi", "wmv", "flv", "mts", "m2ts", "ts", "mpg", "mpeg", "ogv", "3gp", "mp3", "m4a", "wav", "flac", "ogg", "opus", "aac",
];

/// Files handed to the app (CLI args, "Open with", a second launch) before the page is listening.
#[derive(Default)]
struct Pending(Mutex<Vec<String>>);

fn media_paths<I: IntoIterator<Item = String>>(args: I) -> Vec<String> {
    args.into_iter()
        .filter(|a| {
            let p = Path::new(a);
            p.is_file() && p.extension().and_then(|e| e.to_str()).is_some_and(|e| MEDIA_EXT.contains(&e.to_lowercase().as_str()))
        })
        .collect()
}

fn open_paths(app: &AppHandle, paths: Vec<String>) {
    if paths.is_empty() {
        return;
    }
    app.state::<Pending>().0.lock().unwrap().extend(paths.clone());
    let _ = app.emit("open-paths", paths);
}

#[tauri::command]
fn engine_get(engine: State<Engine>) -> EngineState {
    engine.snapshot()
}

#[tauri::command]
fn engine_restart(engine: State<Engine>) {
    engine.restart();
}

/// Debug builds print page errors to the terminal running `tauri dev`.
#[tauri::command]
fn js_log(msg: String) {
    if cfg!(debug_assertions) {
        eprintln!("[webview] {msg}");
    }
}

#[tauri::command]
fn take_pending_paths(pending: State<Pending>) -> Vec<String> {
    std::mem::take(&mut *pending.0.lock().unwrap())
}

/// GUI launches get a minimal PATH; add the usual tool locations so a system ffmpeg,
/// whisper-cli, uv or nvidia-smi is found.
fn extend_path() {
    if cfg!(windows) {
        return;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let cur = std::env::var("PATH").unwrap_or_default();
    let mut parts: Vec<String> = cur.split(':').filter(|s| !s.is_empty()).map(String::from).collect();
    for extra in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/snap/bin", &format!("{home}/.local/bin"), &format!("{home}/.cargo/bin")] {
        if !parts.iter().any(|p| p == extra) {
            parts.push(extra.to_string());
        }
    }
    std::env::set_var("PATH", parts.join(":"));
}

fn create_window(app: &AppHandle) -> tauri::Result<()> {
    let b = WebviewWindowBuilder::new(app, "main", WebviewUrl::default())
        .title("Video Scrubber")
        .inner_size(1440.0, 900.0)
        .min_inner_size(900.0, 600.0)
        .background_color(tauri::window::Color(0xF5, 0xF7, 0xFA, 0xFF))
        .visible(false);
    // macOS keeps native traffic lights over the design's title bar; Windows and Linux draw the
    // design's own caption buttons.
    #[cfg(target_os = "macos")]
    let b = b
        .title_bar_style(tauri::TitleBarStyle::Overlay)
        .hidden_title(true)
        .traffic_light_position(tauri::LogicalPosition::new(14.0, 19.0));
    #[cfg(not(target_os = "macos"))]
    let b = b.decorations(false);
    let w = b.build()?;
    w.show()?;
    Ok(())
}

pub fn run() {
    extend_path();
    let mut builder = tauri::Builder::default();
    #[cfg(any(target_os = "macos", windows, target_os = "linux"))]
    {
        builder = builder.plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            open_paths(app, media_paths(argv.into_iter().skip(1)));
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }));
    }
    let app = builder
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Pending::default())
        .invoke_handler(tauri::generate_handler![engine_get, engine_restart, take_pending_paths, js_log])
        .setup(|app| {
            let handle = app.handle().clone();
            let engine = Engine::new(handle.clone());
            app.manage(engine.clone());
            create_window(&handle)?;
            engine.start();
            app.state::<Pending>().0.lock().unwrap().extend(media_paths(std::env::args().skip(1)));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Video Scrubber");

    app.run(|app, event| match event {
        #[cfg(target_os = "macos")]
        RunEvent::Opened { urls } => {
            let paths = urls.into_iter().filter_map(|u| u.to_file_path().ok()).map(|p| p.to_string_lossy().into_owned());
            open_paths(app, media_paths(paths));
        }
        RunEvent::Exit => app.state::<Engine>().shutdown(),
        _ => {}
    });
}
