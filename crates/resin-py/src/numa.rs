//! Private, calling-thread Linux NUMA memory-policy bindings.

use pyo3::prelude::*;

/// Read the calling thread's allocation policy without changing existing pages.
#[pyfunction]
pub(crate) fn _numa_get_policy() -> PyResult<(i32, Vec<usize>)> {
    #[cfg(target_os = "linux")]
    {
        platform::get_policy().map_err(policy_error)
    }
    #[cfg(not(target_os = "linux"))]
    unsupported()
}

/// Set the calling thread's allocation policy for subsequent allocations.
#[pyfunction]
pub(crate) fn _numa_set_policy(mode: i32, nodes: Vec<usize>) -> PyResult<()> {
    #[cfg(target_os = "linux")]
    {
        platform::set_policy(mode, &nodes).map_err(policy_error)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (mode, nodes);
        unsupported()
    }
}

#[cfg(target_os = "linux")]
fn policy_error(error: std::io::Error) -> PyErr {
    // PyO3's generic io::Error conversion only carries its string, losing errno.
    pyo3::exceptions::PyOSError::new_err((
        error.raw_os_error().unwrap_or(libc::EINVAL),
        error.to_string(),
    ))
}

#[cfg(not(target_os = "linux"))]
fn unsupported<T>() -> PyResult<T> {
    Python::attach(|py| {
        let errno = py.import("errno")?.getattr("ENOSYS")?.extract::<i32>()?;
        Err(pyo3::exceptions::PyOSError::new_err((
            errno,
            "NUMA memory policies require Linux",
        )))
    })
}

#[cfg(all(test, not(target_os = "linux")))]
mod tests {
    use super::*;

    #[test]
    fn non_linux_policy_calls_report_enosys() {
        Python::initialize();
        Python::attach(|py| {
            let expected = py.import("errno").unwrap().getattr("ENOSYS").unwrap();
            for error in [
                _numa_get_policy().unwrap_err(),
                _numa_set_policy(0, Vec::new()).unwrap_err(),
            ] {
                assert!(error.is_instance_of::<pyo3::exceptions::PyOSError>(py));
                assert!(
                    error
                        .value(py)
                        .getattr("errno")
                        .unwrap()
                        .eq(&expected)
                        .unwrap()
                );
            }
        });
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::io;
    use std::ptr;

    type Word = libc::c_ulong;
    const WORD_BITS: usize = Word::BITS as usize;
    const INITIAL_NODE_BITS: usize = 128;
    // Linux configurations use far smaller node masks. Bound both the get retry
    // loop and user-supplied node IDs without assuming the kernel's MAX_NUMNODES.
    const MAX_NODE_BITS: usize = 1 << 20;

    fn encode_nodes(nodes: &[usize]) -> io::Result<Vec<Word>> {
        let Some(&highest) = nodes.iter().max() else {
            return Ok(Vec::new());
        };
        if highest >= MAX_NODE_BITS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("NUMA node ID must be below {MAX_NODE_BITS}"),
            ));
        }
        let mut mask = vec![0; highest / WORD_BITS + 1];
        for &node in nodes {
            mask[node / WORD_BITS] |= 1 << (node % WORD_BITS);
        }
        Ok(mask)
    }

    fn decode_nodes(mask: &[Word]) -> Vec<usize> {
        mask.iter()
            .enumerate()
            .flat_map(|(index, &word)| {
                (0..WORD_BITS)
                    .filter(move |bit| word & (1 << bit) != 0)
                    .map(move |bit| index * WORD_BITS + bit)
            })
            .collect()
    }

    fn read_policy(
        mut read: impl FnMut(&mut i32, &mut [Word], usize) -> io::Result<()>,
    ) -> io::Result<(i32, Vec<usize>)> {
        let mut bits = INITIAL_NODE_BITS;
        loop {
            let mut mode = 0;
            let mut mask = vec![0; bits.div_ceil(WORD_BITS)];
            match read(&mut mode, &mut mask, bits) {
                Ok(()) => return Ok((mode, decode_nodes(&mask))),
                Err(error)
                    if error.raw_os_error() == Some(libc::EINVAL) && bits < MAX_NODE_BITS =>
                {
                    // get_mempolicy rejects masks smaller than MAX_NUMNODES.
                    bits *= 2;
                }
                Err(error) => return Err(error),
            }
        }
    }

    pub(super) fn get_policy() -> io::Result<(i32, Vec<usize>)> {
        read_policy(|mode, mask, bits| {
            // SAFETY: mode and mask are writable for their complete declared
            // sizes; flags=0 queries this thread, so addr is unused and null.
            let result = unsafe {
                libc::syscall(
                    libc::SYS_get_mempolicy,
                    mode as *mut i32,
                    mask.as_mut_ptr(),
                    bits as libc::c_ulong,
                    ptr::null_mut::<libc::c_void>(),
                    0 as libc::c_ulong,
                )
            };
            if result == -1 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        })
    }

    fn write_policy(
        mode: i32,
        nodes: &[usize],
        mut write: impl FnMut(i32, &[Word], usize) -> io::Result<()>,
    ) -> io::Result<()> {
        let mask = encode_nodes(nodes)?;
        // Linux get_nodes() decrements maxnode before reading the mask. Keep
        // the last bit of a complete word, including when restoring a policy.
        let maxnode = if mask.is_empty() {
            0
        } else {
            mask.len() * WORD_BITS + 1
        };
        write(mode, &mask, maxnode)
    }

    pub(super) fn set_policy(mode: i32, nodes: &[usize]) -> io::Result<()> {
        write_policy(mode, nodes, |mode, mask, maxnode| {
            // DEFAULT and LOCAL require an empty mask. A null pointer and zero
            // size also preserve the local form of PREFERRED from get_mempolicy.
            let pointer = if mask.is_empty() {
                ptr::null()
            } else {
                mask.as_ptr()
            };
            // SAFETY: pointer is null with maxnode=0, or readable for all
            // maxnode-1 mask bits. The kernel validates mode and node IDs.
            let result = unsafe {
                libc::syscall(
                    libc::SYS_set_mempolicy,
                    mode as libc::c_int,
                    pointer,
                    maxnode as libc::c_ulong,
                )
            };
            if result == -1 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn node_masks_preserve_sparse_nodes_and_word_boundaries() {
            let nodes = [0, WORD_BITS - 1, WORD_BITS, WORD_BITS * 5 + 2];
            let mask = encode_nodes(&nodes).unwrap();
            assert_eq!(decode_nodes(&mask), nodes);
            assert!(encode_nodes(&[]).unwrap().is_empty());
            assert!(encode_nodes(&[MAX_NODE_BITS]).is_err());
            assert!(encode_nodes(&[usize::MAX]).is_err());
        }

        #[test]
        fn set_policy_preserves_boundary_nodes_with_linux_mask_semantics() {
            let mode = 2 | (1 << 15); // MPOL_BIND | MPOL_F_STATIC_NODES.
            for nodes in [
                vec![],
                vec![0],
                vec![WORD_BITS - 1],
                vec![0, 2 * WORD_BITS - 1],
            ] {
                write_policy(mode, &nodes, |received_mode, mask, maxnode| {
                    // Emulate get_nodes()/get_bitmap(): Linux decrements
                    // maxnode, then clears all bits outside the resulting size.
                    let imported = decode_nodes(mask)
                        .into_iter()
                        .filter(|node| *node < maxnode.saturating_sub(1))
                        .collect::<Vec<_>>();
                    assert_eq!(received_mode, mode);
                    assert_eq!(imported, nodes);
                    if nodes.is_empty() {
                        assert_eq!(maxnode, 0);
                    }
                    Ok(())
                })
                .unwrap();
            }
        }

        #[test]
        fn get_policy_grows_mask_and_preserves_complete_mode() {
            let mut attempts = 0;
            let policy = read_policy(|mode, mask, bits| {
                attempts += 1;
                if bits < 1024 {
                    return Err(io::Error::from_raw_os_error(libc::EINVAL));
                }
                *mode = 2 | (1 << 15); // MPOL_BIND | MPOL_F_STATIC_NODES.
                mask[9] = 1 << 3;
                Ok(())
            })
            .unwrap();
            assert_eq!(attempts, 4);
            assert_eq!(policy, (2 | (1 << 15), vec![9 * WORD_BITS + 3]));
        }

        #[test]
        fn get_policy_preserves_syscall_errors_and_bounds_growth() {
            let error =
                read_policy(|_, _, _| Err(io::Error::from_raw_os_error(libc::EPERM))).unwrap_err();
            assert_eq!(error.raw_os_error(), Some(libc::EPERM));
            let mut last_bits = 0;
            let error = read_policy(|_, _, bits| {
                last_bits = bits;
                Err(io::Error::from_raw_os_error(libc::EINVAL))
            })
            .unwrap_err();
            assert_eq!(last_bits, MAX_NODE_BITS);
            assert_eq!(error.raw_os_error(), Some(libc::EINVAL));
        }
    }
}
