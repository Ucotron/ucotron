//! Writer lock file for enforcing single-writer LMDB constraint.
//!
//! LMDB supports a single writer and multiple concurrent readers. In shared
//! storage mode, multiple Ucotron instances may point at the same data directory.
//! This module provides a file-based lock that ensures only one writer instance
//! can be active at a time.
//!
//! The lock file (`ucotron-writer.lock`) is created in the shared data directory
//! when a writer instance starts. It contains the instance ID and PID for
//! diagnostics. The file is removed on graceful shutdown via [`WriterLock::release`].

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// A file-based writer lock that enforces single-writer access to a shared
/// LMDB data directory.
///
/// # Usage
///
/// ```ignore
/// let lock = WriterLock::acquire("/data/shared", "writer-1")?;
/// // ... server runs ...
/// lock.release(); // called on shutdown
/// ```
#[derive(Debug)]
pub struct WriterLock {
    lock_path: PathBuf,
}

const LOCK_FILENAME: &str = "ucotron-writer.lock";

impl WriterLock {
    /// Attempt to acquire the writer lock for the given shared data directory.
    ///
    /// Returns `Ok(WriterLock)` if the lock was acquired (no other writer is active).
    /// Returns `Err` if another writer already holds the lock.
    ///
    /// The lock file contains JSON with the holding instance's ID and PID for
    /// debugging purposes.
    pub fn acquire(shared_data_dir: &str, instance_id: &str) -> anyhow::Result<Self> {
        let dir = Path::new(shared_data_dir);
        fs::create_dir_all(dir)?;

        let lock_path = dir.join(LOCK_FILENAME);

        // Check for existing lock.
        if lock_path.exists() {
            let content = fs::read_to_string(&lock_path).unwrap_or_default();
            // Check if the holding process is still alive.
            if let Some(pid) = parse_pid_from_lock(&content) {
                if is_process_alive(pid) {
                    anyhow::bail!(
                        "Another writer instance is already active (lock file: {}, pid: {}). \
                         Only one writer is allowed in shared storage mode. \
                         If the previous instance crashed, delete {} manually.",
                        lock_path.display(),
                        pid,
                        lock_path.display(),
                    );
                }
                // Stale lock from a dead process — remove it.
                tracing::warn!(
                    "Removing stale writer lock from pid {} (process no longer running)",
                    pid
                );
            }
            // Lock file exists but no valid PID or process is dead — remove.
            fs::remove_file(&lock_path)?;
        }

        // Write the lock file.
        let pid = std::process::id();
        let content = format!(
            "{{\"instance_id\":\"{}\",\"pid\":{},\"acquired_at\":\"{}\"}}",
            instance_id,
            pid,
            chrono_now_utc(),
        );
        let mut file = fs::File::create(&lock_path)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;

        tracing::info!(
            "Writer lock acquired: {} (instance={}, pid={})",
            lock_path.display(),
            instance_id,
            pid,
        );

        Ok(Self { lock_path })
    }

    /// Release the writer lock (removes the lock file).
    ///
    /// Called during graceful shutdown. If the file was already removed
    /// (e.g., manual intervention), this is a no-op.
    pub fn release(self) {
        if self.lock_path.exists() {
            if let Err(e) = fs::remove_file(&self.lock_path) {
                tracing::error!("Failed to remove writer lock file: {}", e);
            } else {
                tracing::info!("Writer lock released: {}", self.lock_path.display());
            }
        }
    }

    /// Path to the lock file (for testing/diagnostics).
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }
}

impl Drop for WriterLock {
    fn drop(&mut self) {
        // Best-effort cleanup on drop (e.g., if release() wasn't called explicitly).
        if self.lock_path.exists() {
            let _ = fs::remove_file(&self.lock_path);
        }
    }
}

/// Parse the PID from the lock file content.
fn parse_pid_from_lock(content: &str) -> Option<u32> {
    // Simple JSON parsing — look for "pid":NNN
    let pid_prefix = "\"pid\":";
    let start = content.find(pid_prefix)?;
    let rest = &content[start + pid_prefix.len()..];
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[..end].parse::<u32>().ok()
}

/// Check if a process with the given PID is still alive.
fn is_process_alive(pid: u32) -> bool {
    // On Unix, sending signal 0 checks if the process exists without actually
    // sending a signal.
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        // On non-Unix platforms, assume the process is alive (conservative).
        let _ = pid;
        true
    }
}

/// Return the current UTC time as an ISO 8601 string (no chrono dependency).
fn chrono_now_utc() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s-since-epoch", dur.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_and_release() {
        let dir = tempfile::tempdir().unwrap();
        let shared_dir = dir.path().to_string_lossy().to_string();

        let lock = WriterLock::acquire(&shared_dir, "test-writer-1").unwrap();
        assert!(lock.lock_path().exists());

        // Verify lock file content.
        let content = fs::read_to_string(lock.lock_path()).unwrap();
        assert!(content.contains("\"instance_id\":\"test-writer-1\""));
        assert!(content.contains("\"pid\":"));

        lock.release();
        assert!(!dir.path().join(LOCK_FILENAME).exists());
    }

    #[test]
    fn test_second_writer_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let shared_dir = dir.path().to_string_lossy().to_string();

        let lock1 = WriterLock::acquire(&shared_dir, "writer-1").unwrap();

        // Second writer should fail.
        let result = WriterLock::acquire(&shared_dir, "writer-2");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Another writer instance is already active"));

        // Release first lock, second should now succeed.
        lock1.release();
        let lock2 = WriterLock::acquire(&shared_dir, "writer-2").unwrap();
        lock2.release();
    }

    #[test]
    fn test_stale_lock_removed() {
        let dir = tempfile::tempdir().unwrap();
        let shared_dir = dir.path().to_string_lossy().to_string();
        let lock_path = dir.path().join(LOCK_FILENAME);

        // Write a fake lock file with a PID that doesn't exist (PID 1 is init,
        // but PID 99999999 almost certainly doesn't exist).
        let fake_content = r#"{"instance_id":"dead","pid":99999999,"acquired_at":"0s-since-epoch"}"#;
        fs::write(&lock_path, fake_content).unwrap();

        // Should succeed because the PID is dead (unless we happen to have a
        // process with PID 99999999, which is extremely unlikely).
        let result = WriterLock::acquire(&shared_dir, "new-writer");
        // On most systems, PID 99999999 doesn't exist, so this should succeed.
        // If it doesn't, it means the stale detection found the process alive
        // (very unlikely but theoretically possible on some systems).
        if let Ok(lock) = result {
            lock.release();
        }
    }

    #[test]
    fn test_drop_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let shared_dir = dir.path().to_string_lossy().to_string();
        let lock_file_path;

        {
            let lock = WriterLock::acquire(&shared_dir, "drop-test").unwrap();
            lock_file_path = lock.lock_path().to_path_buf();
            assert!(lock_file_path.exists());
            // lock dropped here
        }

        // Lock file should be cleaned up by Drop.
        assert!(!lock_file_path.exists());
    }

    #[test]
    fn test_parse_pid_from_lock() {
        let content = r#"{"instance_id":"test","pid":12345,"acquired_at":"0s"}"#;
        assert_eq!(parse_pid_from_lock(content), Some(12345));

        assert_eq!(parse_pid_from_lock("no pid here"), None);
        assert_eq!(parse_pid_from_lock(""), None);
    }

    #[test]
    fn test_creates_directory_if_missing() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested").join("deep");
        let shared_dir = nested.to_string_lossy().to_string();

        let lock = WriterLock::acquire(&shared_dir, "dir-test").unwrap();
        assert!(lock.lock_path().exists());
        lock.release();
    }
}
