//! PSD channel extraction and PackBits row encoding.
//!
//! GIMP reference: PackBits packet construction:
//! https://github.com/GNOME/gimp/blob/758fb4ed995bbb339282d3f777089a33f0a391b8/plug-ins/file-psd/psd-util.c#L907-L1034
//! GIMP reference: alpha-first layer channels and per-row length tables:
//! https://github.com/GNOME/gimp/blob/758fb4ed995bbb339282d3f777089a33f0a391b8/plug-ins/file-psd/psd-export.c#L1772-L1974

use image::RgbaImage;

use crate::error::PsdExportError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelId {
    Red,
    Green,
    Blue,
    Alpha,
}

impl ChannelId {
    pub fn psd_id(self) -> i16 {
        match self {
            Self::Red => 0,
            Self::Green => 1,
            Self::Blue => 2,
            Self::Alpha => -1,
        }
    }

    fn rgba_offset(self) -> usize {
        match self {
            Self::Red => 0,
            Self::Green => 1,
            Self::Blue => 2,
            Self::Alpha => 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncodedChannel {
    pub channel_id: i16,
    pub data: Vec<u8>,
}

pub fn encode_image_rle(
    image: &RgbaImage,
    channels: &[ChannelId],
    layer_name: &str,
) -> Result<Vec<EncodedChannel>, PsdExportError> {
    let mut encoded = Vec::with_capacity(channels.len());

    for channel in channels {
        let mut lengths = Vec::with_capacity(image.height() as usize);
        let mut data = Vec::new();
        let width = image.width() as usize;
        let offset = channel.rgba_offset();

        for y in 0..image.height() {
            let start = data.len();
            let mut row = Vec::with_capacity(width);
            for x in 0..image.width() {
                row.push(image.get_pixel(x, y).0[offset]);
            }
            encode_row(&row, &mut data);
            let row_length = data.len() - start;
            if row_length > u16::MAX as usize {
                return Err(PsdExportError::InvalidChannelEncoding {
                    layer: layer_name.to_string(),
                    row: y as usize,
                    length: row_length,
                });
            }
            lengths.push(row_length as u16);
        }

        let mut out = Vec::with_capacity(lengths.len() * 2 + data.len());
        for length in lengths {
            out.extend_from_slice(&length.to_be_bytes());
        }
        out.extend_from_slice(&data);
        encoded.push(EncodedChannel {
            channel_id: channel.psd_id(),
            data: out,
        });
    }

    Ok(encoded)
}

fn encode_row(row: &[u8], out: &mut Vec<u8>) {
    let mut i = 0usize;
    while i < row.len() {
        let run_len = repeated_run_len(row, i);
        if run_len >= 3 {
            let chunk = run_len.min(128);
            out.push((1i16 - chunk as i16) as u8);
            out.push(row[i]);
            i += chunk;
            continue;
        }

        let literal_start = i;
        let mut literal_len = 0usize;
        while i < row.len() && literal_len < 128 {
            let next_run = repeated_run_len(row, i);
            if next_run >= 3 {
                break;
            }
            i += 1;
            literal_len += 1;
        }

        out.push((literal_len - 1) as u8);
        out.extend_from_slice(&row[literal_start..literal_start + literal_len]);
    }
}

fn repeated_run_len(row: &[u8], start: usize) -> usize {
    let value = row[start];
    let mut len = 1usize;
    while start + len < row.len() && row[start + len] == value && len < 128 {
        len += 1;
    }
    len
}

#[cfg(test)]
mod tests {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::Cell,
    };

    use image::{Rgba, RgbaImage};

    use super::{ChannelId, encode_image_rle, encode_row};

    thread_local! {
        static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    }

    /// Counts heap allocations made by the current thread.
    struct CountingAllocator;

    // SAFETY: every call forwards to the system allocator unchanged.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    const ALL: [ChannelId; 4] = [
        ChannelId::Alpha,
        ChannelId::Red,
        ChannelId::Green,
        ChannelId::Blue,
    ];

    /// The encoder before row extraction moved to slices: one `get_pixel` per
    /// byte, then the row table copied in front of the payload.
    fn reference_rle(image: &RgbaImage, channels: &[ChannelId]) -> Vec<(i16, Vec<u8>)> {
        channels
            .iter()
            .map(|channel| {
                let mut lengths = Vec::new();
                let mut data = Vec::new();
                for y in 0..image.height() {
                    let start = data.len();
                    let row = (0..image.width())
                        .map(|x| image.get_pixel(x, y).0[channel.rgba_offset()])
                        .collect::<Vec<_>>();
                    encode_row(&row, &mut data);
                    lengths.push((data.len() - start) as u16);
                }
                let mut out = Vec::new();
                for length in lengths {
                    out.extend_from_slice(&length.to_be_bytes());
                }
                out.extend_from_slice(&data);
                (channel.psd_id(), out)
            })
            .collect()
    }

    fn noise(width: u32, height: u32, seed: u32) -> RgbaImage {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        RgbaImage::from_fn(width, height, |x, y| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let [a, b, c, d] = state.to_le_bytes();
            // Mix literal noise with flat runs so both packet kinds and the
            // 128-byte packet limit appear in the same rows.
            if (x / 3 + y) % 4 == 0 || x > 140 {
                Rgba([7, 7, 7, 0])
            } else {
                Rgba([a, b & 0xf0, c, d | 1])
            }
        })
    }

    #[test]
    fn slice_rows_encode_like_per_pixel_reads() {
        let mut images = Vec::new();
        for width in [1, 2, 3, 127, 128, 129, 130, 255, 257, 301] {
            for height in [1, 2, 5] {
                images.push(noise(width, height, width * 31 + height));
            }
        }
        images.push(RgbaImage::from_pixel(300, 3, Rgba([1, 2, 3, 4])));
        images.extend([RgbaImage::new(0, 3), RgbaImage::new(3, 0)]);
        images.push(RgbaImage::from_fn(131, 4, |x, y| {
            Rgba([
                x as u8,
                (x / 2) as u8,
                y as u8,
                if x % 2 == 0 { 0 } else { 255 },
            ])
        }));

        for image in &images {
            for channels in [
                &ALL[..],
                &[ChannelId::Red, ChannelId::Green, ChannelId::Blue][..],
            ] {
                let encoded = encode_image_rle(image, channels, "layer")
                    .expect("encode")
                    .into_iter()
                    .map(|channel| (channel.channel_id, channel.data))
                    .collect::<Vec<_>>();
                assert_eq!(
                    encoded,
                    reference_rle(image, channels),
                    "{}x{} image",
                    image.width(),
                    image.height()
                );
            }
        }
    }

    #[test]
    fn row_extraction_does_not_allocate_per_row() {
        let image = noise(64, 256, 3);
        let before = ALLOCATIONS.with(Cell::get);
        let encoded = encode_image_rle(&image, &[ChannelId::Red], "layer").expect("encode");
        let allocations = ALLOCATIONS.with(Cell::get) - before;
        drop(encoded);
        assert!(
            allocations < 32,
            "{allocations} allocations to encode one {}-row channel",
            image.height()
        );
    }

    #[test]
    fn packbits_repeated_rows_encode_with_short_repeat_packets() {
        let mut image = RgbaImage::new(4, 1);
        for x in 0..4 {
            image.put_pixel(x, 0, Rgba([10, 0, 0, 255]));
        }

        let channels = encode_image_rle(&image, &[ChannelId::Red], "row").expect("encode");
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].data, vec![0, 2, 253, 10]);
    }

    #[test]
    fn packbits_literal_rows_keep_original_order() {
        let mut image = RgbaImage::new(4, 1);
        let values = [1u8, 2, 3, 4];
        for (x, value) in values.into_iter().enumerate() {
            image.put_pixel(x as u32, 0, Rgba([value, 0, 0, 255]));
        }

        let channels = encode_image_rle(&image, &[ChannelId::Red], "row").expect("encode");
        assert_eq!(channels[0].data, vec![0, 5, 3, 1, 2, 3, 4]);
    }
}
