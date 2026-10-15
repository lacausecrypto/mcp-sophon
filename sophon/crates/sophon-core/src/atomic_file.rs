//! Crash-safe replacement of a single local file without truncating its
//! previous contents before the new bytes are fully written.
//!
//! Every replacement is staged in the destination directory (same filesystem),
//! synced, and renamed over the destination. The rename is the commit point.
//! This helper does **not** coordinate multiple distinct files or processes
//! writing to the same destination.
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Replace a single file atomically, preserving old bytes if staging or the
/// rename fails. The caller must serialize concurrent writes to the same path.
pub fn replace(path: &Path, contents: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let basename = path.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "destination has no filename")
    })?;
    // A rename over an existing symlink would replace the link rather than
    // updating its target. Refuse that surprising and potentially unsafe
    // operation. Preserve previous file permissions whenever possible.
    let previous_permissions = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "atomic replacement does not follow destination symlinks",
            ));
        }
        Ok(metadata) if metadata.permissions().readonly() => {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "atomic replacement refused for a read-only destination",
            ));
        }
        Ok(metadata) => Some(metadata.permissions()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => None,
        Err(err) => return Err(err),
    };

    for _ in 0..16 {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let temp_name = format!(
            ".{}.sophon-{}-{}.tmp",
            basename.to_string_lossy(),
            std::process::id(),
            id
        );
        let staged = parent.join(temp_name);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        // Persistent memory and graph data may be private. Never create a
        // replacement with world-readable default permissions on Unix.
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = match options.open(&staged) {
            Ok(file) => file,
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        };

        let result = (|| -> io::Result<()> {
            file.write_all(contents)?;
            if let Some(permissions) = &previous_permissions {
                file.set_permissions(permissions.clone())?;
            }
            file.sync_all()?;
            drop(file);
            fs::rename(&staged, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&staged);
        } else {
            // Once rename succeeds the new file is already visible. Do not
            // report a failure after commit if directory fsync is unsupported
            // (e.g. on some container filesystems).
            #[cfg(unix)]
            {
                if let Ok(dir) = fs::File::open(parent) {
                    let _ = dir.sync_all();
                }
            }
        }
        return result;
    }

    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique staging file",
    ))
}

#[cfg(test)]
mod tests {
    use super::replace;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "sophon-atomic-tests-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn replaces_existing_content_without_a_temp_left_behind() {
        let dir = fixture();
        let path = dir.join("state.json");
        fs::write(&path, b"old").unwrap();
        replace(&path, b"new").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn new_files_are_private_and_replacements_preserve_prior_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = fixture();
        let path = dir.join("private.json");
        replace(&path, b"private").unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );

        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        replace(&path, b"updated").unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn respects_a_read_only_destination() {
        use std::os::unix::fs::PermissionsExt;
        let dir = fixture();
        let path = dir.join("locked.json");
        fs::write(&path, b"original").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o400)).unwrap();
        assert!(replace(&path, b"new").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original");
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symlink_destination_is_not_replaced() {
        use std::os::unix::fs::symlink;
        let dir = fixture();
        let target = dir.join("target.json");
        let link = dir.join("alias.json");
        fs::write(&target, b"protected").unwrap();
        symlink(&target, &link).unwrap();
        assert!(replace(&link, b"overwritten").is_err());
        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(&target).unwrap(), b"protected");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failure_preserves_existing_directory_and_cleans_staging_file() {
        let dir = fixture();
        let target = dir.join("state.json");
        fs::create_dir(&target).unwrap();
        assert!(replace(&target, b"new").is_err());
        assert!(target.is_dir());
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_dir_all(dir).unwrap();
    }
}
