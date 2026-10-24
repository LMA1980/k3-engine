//! Shared test harness and opaque-box IPC/SHM primitives for US-05 E2E integration tests.
//!
//! Provides portable shared memory arenas (Linux `memfd_create` + macOS anonymous shm),
//! SCM_RIGHTS file descriptor passing over Unix Domain Sockets,
//! zerocopy memory layout definitions, and child process supervision helpers.

#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_qualifications,
    unreachable_pub,
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::undocumented_unsafe_blocks,
    clippy::const_is_empty
)]

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

pub const ARENA_MAGIC: [u8; 4] = *b"K3KV";
pub const ARENA_VERSION: u32 = 1;
pub const BLOCK_SIZE_TOKENS: u32 = 16;
pub const CACHELINE_ALIGNMENT: usize = 64;

// Block tag bitmask constants
pub const TAG_PURGE: u32 = 1 << 0;
pub const TAG_PERSIST: u32 = 1 << 1;
pub const TAG_VOLATILE: u32 = 1 << 2;

// Exit code constants per ADR-018, ADR-025, ADR-041, AGENTS.md
pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_INVALID_ARGUMENTS: i32 = 2;
pub const EXIT_CAPABILITY_MISMATCH: i32 = 10;
pub const EXIT_INSUFFICIENT_MEMLOCK: i32 = 11;
pub const EXIT_CONTEXT_LENGTH_EXCEEDED: i32 = 12;
pub const EXIT_WORKER_FAULT: i32 = 13;
pub const EXIT_ADMISSION_REFUSED: i32 = 14;

// Core affinity masks for SpacemiT K3
pub const X100_CORES_MASK: u64 = 0x00FF; // Cluster 0: cores 0..7
pub const A100_CORES_MASK: u64 = 0xFF00; // Cluster 1: cores 8..15

static UNIQUE_SEQ: AtomicU64 = AtomicU64::new(1);

pub fn next_seq() -> u64 {
    let prev = UNIQUE_SEQ.fetch_add(1, Ordering::Relaxed);
    prev.wrapping_mul(6364136223846793005).wrapping_add(1)
}

/// Fixed 128-byte header aligned to cacheline boundaries at offset 0 of shared memory arena.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub total_arena_bytes: u64,
    pub layer_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub n_ctx: u32,
    pub block_size: u32,
    pub num_blocks: u32,
    pub meta_table_offset: u64,
    pub meta_table_bytes: u64,
    pub token_slots_offset: u64,
    pub token_slots_bytes: u64,
    pub kv_blocks_offset: u64,
    pub kv_blocks_bytes: u64,
    pub output_slabs_offset: u64,
    pub output_slabs_bytes: u64,
    pub active_slots: u32,
    pub generation_counter: u64,
    pub reserved: [u8; 28],
}

impl Default for ArenaHeader {
    fn default() -> Self {
        Self {
            magic: ARENA_MAGIC,
            version: ARENA_VERSION,
            total_arena_bytes: 0,
            layer_count: 2,
            head_count_kv: 2,
            head_dim: 64,
            n_ctx: 64,
            block_size: BLOCK_SIZE_TOKENS,
            num_blocks: 4,
            meta_table_offset: 128,
            meta_table_bytes: 4 * 64,
            token_slots_offset: 128 + 4 * 64,
            token_slots_bytes: 64 * 4 + 16,
            kv_blocks_offset: 1024,
            kv_blocks_bytes: 4 * 16 * 2 * 2 * 64 * 4,
            output_slabs_offset: 4096,
            output_slabs_bytes: 32000 * 4 + 64,
            active_slots: 1,
            generation_counter: 0,
            reserved: [0; 28],
        }
    }
}

/// 64-byte aligned metadata entry for a 16-token KV cache block.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBlockMeta {
    pub block_id: u32,
    pub tag: u32,
    pub project_id: u64,
    pub refcount: u32,
    pub committed_tokens: u32,
    pub session_id_hash: u64,
    pub last_access_timestamp: u64,
    pub sampler_rng_state: [u64; 4],
    pub padding: [u8; 16],
}

impl Default for KvBlockMeta {
    fn default() -> Self {
        Self {
            block_id: 0,
            tag: TAG_VOLATILE,
            project_id: 1,
            refcount: 1,
            committed_tokens: 0,
            session_id_hash: 0,
            last_access_timestamp: 0,
            sampler_rng_state: [42, 0, 0, 0],
            padding: [0; 16],
        }
    }
}

/// Header preceding input token slot in shared memory.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenInputHeader {
    pub slot_idx: u32,
    pub token_count: u32,
    pub start_position: u32,
    pub flags: u32,
}

/// Header preceding output logit slab in shared memory.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputSlabHeader {
    pub request_id: u64,
    pub position: u32,
    pub vocab_size: u32,
    pub status: u32, // 0 = idle, 1 = written, 2 = read
    pub reserved: u32,
}

/// Portable shared memory arena abstraction.
/// Backed by Linux `memfd_create` or macOS `shm_open` / unlinked tempfile.
pub struct SharedMemoryFixture {
    fd: OwnedFd,
    ptr: NonNull<u8>,
    len: usize,
    name: String,
    path: Option<PathBuf>,
}

unsafe impl Send for SharedMemoryFixture {}
unsafe impl Sync for SharedMemoryFixture {}

impl SharedMemoryFixture {
    pub fn create(name: &str, len: usize) -> io::Result<Self> {
        let page_size = 4096;
        let aligned_len = (len + page_size - 1) & !(page_size - 1);
        let aligned_len = aligned_len.max(page_size);

        #[cfg(target_os = "linux")]
        {
            let cname = std::ffi::CString::new(name).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            // MFD_CLOEXEC = 1, MFD_ALLOW_SEALING = 2
            let raw_fd = unsafe { libc::syscall(libc::SYS_memfd_create, cname.as_ptr(), 1 | 2) as i32 };
            if raw_fd >= 0 {
                let ret = unsafe { libc::ftruncate(raw_fd, aligned_len as libc::off_t) };
                if ret == 0 {
                    let ptr = unsafe {
                        libc::mmap(
                            std::ptr::null_mut(),
                            aligned_len,
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_SHARED,
                            raw_fd,
                            0,
                        )
                    };
                    if ptr != libc::MAP_FAILED {
                        let owned_fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
                        return Ok(Self {
                            fd: owned_fd,
                            ptr: NonNull::new(ptr as *mut u8).expect("valid pointer"),
                            len: aligned_len,
                            name: name.to_string(),
                            path: None,
                        });
                    }
                }
                unsafe { libc::close(raw_fd); }
            }
        }

        // Portable anonymous tempfile fallback (supports macOS & unprivileged Linux)
        let temp_path = std::env::temp_dir().join(format!("k3_shm_{}_{}_{}.shm", name, std::process::id(), next_seq()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;
        file.set_len(aligned_len as u64)?;

        let raw_fd = file.as_raw_fd();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                aligned_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                raw_fd,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // Zero-initialize mapped area
        unsafe {
            std::ptr::write_bytes(ptr as *mut u8, 0, aligned_len);
        }

        let owned_fd = file.into();
        Ok(Self {
            fd: owned_fd,
            ptr: NonNull::new(ptr as *mut u8).expect("valid pointer"),
            len: aligned_len,
            name: name.to_string(),
            path: Some(temp_path),
        })
    }

    pub fn from_fd(owned_fd: OwnedFd, len: usize, name: &str) -> io::Result<Self> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                owned_fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            fd: owned_fd,
            ptr: NonNull::new(ptr as *mut u8).expect("valid pointer"),
            len,
            name: name.to_string(),
            path: None,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    pub fn duplicate_fd(&self) -> io::Result<OwnedFd> {
        let dup_raw = unsafe { libc::dup(self.fd.as_raw_fd()) };
        if dup_raw < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(unsafe { OwnedFd::from_raw_fd(dup_raw) })
        }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn header(&self) -> &ArenaHeader {
        assert!(self.len >= std::mem::size_of::<ArenaHeader>());
        unsafe { &*(self.ptr.as_ptr() as *const ArenaHeader) }
    }

    pub fn header_mut(&mut self) -> &mut ArenaHeader {
        assert!(self.len >= std::mem::size_of::<ArenaHeader>());
        unsafe { &mut *(self.ptr.as_ptr() as *mut ArenaHeader) }
    }

    pub fn block_meta(&self, block_idx: u32) -> Option<&KvBlockMeta> {
        let offset = self.header().meta_table_offset as usize;
        let entry_size = std::mem::size_of::<KvBlockMeta>();
        let target = offset + (block_idx as usize) * entry_size;
        if target + entry_size <= self.len {
            unsafe { Some(&*(self.ptr.as_ptr().add(target) as *const KvBlockMeta)) }
        } else {
            None
        }
    }

    pub fn block_meta_mut(&mut self, block_idx: u32) -> Option<&mut KvBlockMeta> {
        let offset = self.header().meta_table_offset as usize;
        let entry_size = std::mem::size_of::<KvBlockMeta>();
        let target = offset + (block_idx as usize) * entry_size;
        if target + entry_size <= self.len {
            unsafe { Some(&mut *(self.ptr.as_ptr().add(target) as *mut KvBlockMeta)) }
        } else {
            None
        }
    }

    pub fn write_tokens(&mut self, tokens: &[u32]) -> io::Result<()> {
        let offset = self.header().token_slots_offset as usize;
        let hdr_size = std::mem::size_of::<TokenInputHeader>();
        let payload_size = std::mem::size_of_val(tokens);
        if offset + hdr_size + payload_size > self.len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Token buffer exceeds arena bounds"));
        }
        let hdr = TokenInputHeader {
            slot_idx: 0,
            token_count: tokens.len() as u32,
            start_position: 0,
            flags: 0,
        };
        unsafe {
            let hdr_ptr = self.ptr.as_ptr().add(offset) as *mut TokenInputHeader;
            *hdr_ptr = hdr;
            let tokens_ptr = self.ptr.as_ptr().add(offset + hdr_size) as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), tokens_ptr, tokens.len());
        }
        Ok(())
    }

    pub fn read_tokens(&self) -> Vec<u32> {
        let offset = self.header().token_slots_offset as usize;
        let hdr_size = std::mem::size_of::<TokenInputHeader>();
        if offset + hdr_size > self.len {
            return Vec::new();
        }
        unsafe {
            let hdr = *(self.ptr.as_ptr().add(offset) as *const TokenInputHeader);
            let tokens_ptr = self.ptr.as_ptr().add(offset + hdr_size) as *const u32;
            let mut out = vec![0_u32; hdr.token_count as usize];
            std::ptr::copy_nonoverlapping(tokens_ptr, out.as_mut_ptr(), hdr.token_count as usize);
            out
        }
    }

    pub fn write_logits(&mut self, req_id: u64, logits: &[f32]) -> io::Result<()> {
        let offset = self.header().output_slabs_offset as usize;
        let hdr_size = std::mem::size_of::<OutputSlabHeader>();
        let payload_size = std::mem::size_of_val(logits);
        if offset + hdr_size + payload_size > self.len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Logit buffer exceeds arena bounds"));
        }
        let hdr = OutputSlabHeader {
            request_id: req_id,
            position: 0,
            vocab_size: logits.len() as u32,
            status: 1, // written
            reserved: 0,
        };
        unsafe {
            let hdr_ptr = self.ptr.as_ptr().add(offset) as *mut OutputSlabHeader;
            *hdr_ptr = hdr;
            let logits_ptr = self.ptr.as_ptr().add(offset + hdr_size) as *mut f32;
            std::ptr::copy_nonoverlapping(logits.as_ptr(), logits_ptr, logits.len());
        }
        Ok(())
    }

    pub fn read_logits(&self) -> (OutputSlabHeader, Vec<f32>) {
        let offset = self.header().output_slabs_offset as usize;
        let hdr_size = std::mem::size_of::<OutputSlabHeader>();
        if offset + hdr_size > self.len {
            return (OutputSlabHeader { request_id: 0, position: 0, vocab_size: 0, status: 0, reserved: 0 }, Vec::new());
        }
        unsafe {
            let hdr = *(self.ptr.as_ptr().add(offset) as *const OutputSlabHeader);
            let logits_ptr = self.ptr.as_ptr().add(offset + hdr_size) as *const f32;
            let mut out = vec![0_f32; hdr.vocab_size as usize];
            std::ptr::copy_nonoverlapping(logits_ptr, out.as_mut_ptr(), hdr.vocab_size as usize);
            (hdr, out)
        }
    }
}

impl Drop for SharedMemoryFixture {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len);
        }
        if let Some(path) = &self.path {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Socket communication helpers with SCM_RIGHTS FD transfer support.
pub struct SocketIpcHelper;

impl SocketIpcHelper {
    /// Creates a connected bidirectional Unix stream socket pair.
    pub fn create_socket_pair() -> io::Result<(OwnedFd, OwnedFd)> {
        let mut fds = [0_i32; 2];
        let ret = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
        unsafe {
            Ok((OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])))
        }
    }

    /// Sends a length-prefixed payload over UDS, optionally passing a file descriptor via SCM_RIGHTS.
    pub fn send_message_with_fd(sock: RawFd, payload: &[u8], fd_to_pass: Option<RawFd>) -> io::Result<usize> {
        let len_prefix = (payload.len() as u32).to_be_bytes();
        let mut full_payload = Vec::with_capacity(4 + payload.len());
        full_payload.extend_from_slice(&len_prefix);
        full_payload.extend_from_slice(payload);

        let mut iov = libc::iovec {
            iov_base: full_payload.as_ptr() as *mut libc::c_void,
            iov_len: full_payload.len(),
        };

        let mut cmsg_buffer = [0_u8; 128];
        let (cmsg_ptr, cmsg_len) = if let Some(fd) = fd_to_pass {
            unsafe {
                let space = libc::CMSG_SPACE(std::mem::size_of::<libc::c_int>() as u32) as usize;
                assert!(space <= cmsg_buffer.len());
                let mut dummy_hdr: libc::msghdr = std::mem::zeroed();
                dummy_hdr.msg_control = cmsg_buffer.as_mut_ptr() as *mut libc::c_void;
                dummy_hdr.msg_controllen = space as _;

                let cmsg = libc::CMSG_FIRSTHDR(&dummy_hdr);
                if cmsg.is_null() {
                    return Err(io::Error::other("CMSG_FIRSTHDR failed"));
                }
                (*cmsg).cmsg_level = libc::SOL_SOCKET;
                (*cmsg).cmsg_type = libc::SCM_RIGHTS;
                (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as u32) as _;

                let data_ptr = libc::CMSG_DATA(cmsg) as *mut libc::c_int;
                *data_ptr = fd;

                (cmsg_buffer.as_mut_ptr() as *mut libc::c_void, space)
            }
        } else {
            (std::ptr::null_mut(), 0)
        };

        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_ptr;
        msg.msg_controllen = cmsg_len as _;

        let sent = unsafe { libc::sendmsg(sock, &msg, 0) };
        if sent < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(sent as usize)
        }
    }

    /// Receives a length-prefixed payload over UDS along with any SCM_RIGHTS file descriptor.
    pub fn recv_message_with_fd(sock: RawFd) -> io::Result<(Vec<u8>, Option<OwnedFd>)> {
        let mut len_buf = [0_u8; 4];
        let mut iov = libc::iovec {
            iov_base: len_buf.as_mut_ptr() as *mut libc::c_void,
            iov_len: 4,
        };

        let mut cmsg_buffer = [0_u8; 128];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buffer.as_mut_ptr() as *mut libc::c_void;
        msg.msg_controllen = cmsg_buffer.len() as _;

        let bytes_read = unsafe { libc::recvmsg(sock, &mut msg, 0) };
        if bytes_read <= 0 {
            if bytes_read == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Socket closed by peer"));
            }
            return Err(io::Error::last_os_error());
        }

        let mut received_fd: Option<OwnedFd> = None;
        unsafe {
            let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
            while !cmsg.is_null() {
                if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                    let fd_ptr = libc::CMSG_DATA(cmsg) as *const libc::c_int;
                    let raw_fd = *fd_ptr;
                    if raw_fd >= 0 {
                        received_fd = Some(OwnedFd::from_raw_fd(raw_fd));
                        break;
                    }
                }
                cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
            }
        }

        let payload_len = u32::from_be_bytes(len_buf) as usize;
        let mut payload = vec![0_u8; payload_len];
        if payload_len > 0 {
            let mut total = 0;
            while total < payload_len {
                let n = unsafe {
                    libc::recv(
                        sock,
                        payload.as_mut_ptr().add(total) as *mut libc::c_void,
                        payload_len - total,
                        0,
                    )
                };
                if n <= 0 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated payload"));
                }
                total += n as usize;
            }
        }

        Ok((payload, received_fd))
    }
}
