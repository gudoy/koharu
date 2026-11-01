//! Bounded encoded-image storage, header validation, and on-demand decoding.

use std::{collections::HashMap, io::Cursor, sync::Arc};

use koharu_scene::BlobId;

use crate::{Error, Result};

pub(crate) const DEFAULT_IMAGE_CACHE_BYTES: usize = 512 * 1024 * 1024;

/// A blob and the dimensions read from its header. Presentation ships the
/// encoded bytes and the browser decodes them, so RGBA is decoded only when a
/// native rasterization asks for it.
pub(crate) struct EncodedImage {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) encoded: Arc<[u8]>,
}

impl EncodedImage {
    fn byte_len(&self) -> usize {
        self.encoded.len()
    }
}

struct CachedImage {
    image: Arc<EncodedImage>,
    last_used: u64,
}

pub(crate) struct ImageCache {
    entries: HashMap<BlobId, CachedImage>,
    max_bytes: usize,
    bytes: usize,
    clock: u64,
}

impl ImageCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            max_bytes: DEFAULT_IMAGE_CACHE_BYTES,
            bytes: 0,
            clock: 0,
        }
    }

    #[cfg(test)]
    fn with_limit(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ..Self::new()
        }
    }

    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    pub(crate) fn get(&mut self, blob: BlobId) -> Option<Arc<EncodedImage>> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(&blob)?;
        entry.last_used = self.clock;
        Some(entry.image.clone())
    }

    pub(crate) fn insert(&mut self, blob: BlobId, image: Arc<EncodedImage>) {
        self.clock = self.clock.wrapping_add(1);
        if let Some(previous) = self.entries.remove(&blob) {
            self.bytes = self.bytes.saturating_sub(previous.image.byte_len());
        }
        if image.byte_len() > self.max_bytes {
            return;
        }
        self.bytes = self.bytes.saturating_add(image.byte_len());
        self.entries.insert(
            blob,
            CachedImage {
                image,
                last_used: self.clock,
            },
        );
        while self.bytes > self.max_bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(blob, _)| *blob)
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.bytes = self.bytes.saturating_sub(removed.image.byte_len());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_larger_than_the_budget_is_not_cached() {
        let blob = BlobId::for_bytes(b"oversized");
        let image = Arc::new(EncodedImage {
            width: 2,
            height: 1,
            encoded: Arc::from([0_u8; 8]),
        });
        let mut cache = ImageCache::with_limit(4);
        cache.insert(blob, image);
        assert!(cache.get(blob).is_none());
        assert_eq!(cache.bytes, 0);
    }
}

/// Reads only the header: an unknown signature or unreadable dimensions still
/// fail the render, and node construction checks the size against the scene.
pub(crate) fn probe(blob: BlobId, bytes: Arc<[u8]>) -> Result<(BlobId, Arc<EncodedImage>)> {
    let (width, height) = image::ImageReader::new(Cursor::new(bytes.as_ref()))
        .with_guessed_format()
        .map_err(|source| Error::Image {
            blob,
            source: source.into(),
        })?
        .into_dimensions()
        .map_err(|source| Error::Image { blob, source })?;
    Ok((
        blob,
        Arc::new(EncodedImage {
            width,
            height,
            encoded: bytes,
        }),
    ))
}

/// Decodes straight RGBA for native rasterization.
pub(crate) fn decode(blob: BlobId, encoded: &[u8], size: (u32, u32)) -> Result<Arc<[u8]>> {
    let decoded = image::load_from_memory(encoded)
        .map_err(|source| Error::Image { blob, source })?
        .into_rgba8();
    if decoded.dimensions() != size {
        return Err(Error::invalid(format!(
            "blob {blob} decoded as {}x{}, expected {}x{}",
            decoded.width(),
            decoded.height(),
            size.0,
            size.1
        )));
    }
    Ok(Arc::from(decoded.into_raw()))
}
