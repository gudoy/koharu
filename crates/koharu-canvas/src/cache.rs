use std::collections::{HashMap, HashSet};

use koharu_rasterizer::{LayerId, PreparedFrameManifest, ResourceId};

const RESOURCE_CACHE_BUDGET: u64 = 512 * 1024 * 1024;
const MAX_CACHED_RESOURCES: usize = 1_024;

/// Prepared resources held in CPU and GPU memory.
pub(crate) trait ResidentResources {
    fn resident_bytes(&self) -> u64;
    fn release(&mut self, resource: ResourceId);
}

/// Manifests of shown and prefetched pages, kept so a page switch can activate without a fetch.
#[derive(Default)]
pub(crate) struct PageManifests {
    entries: HashMap<LayerId, CachedManifest>,
    clock: u64,
}

struct CachedManifest {
    manifest: PreparedFrameManifest,
    used: u64,
}

impl PageManifests {
    pub(crate) fn get(&self, page: &LayerId) -> Option<&PreparedFrameManifest> {
        self.entries.get(page).map(|entry| &entry.manifest)
    }

    pub(crate) fn insert(&mut self, page: LayerId, manifest: PreparedFrameManifest) {
        self.clock = self.clock.wrapping_add(1);
        self.entries.insert(
            page,
            CachedManifest {
                manifest,
                used: self.clock,
            },
        );
    }

    fn resources(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.entries.values().flat_map(|entry| {
            entry
                .manifest
                .required_resources()
                .iter()
                .map(|resource| resource.id)
        })
    }

    /// Drops the least recently used page that holds a resource outside `pinned`. Dropping a
    /// page whose resources are all pinned, such as the active one, would free nothing.
    fn drop_least_recent(&mut self, pinned: &HashSet<ResourceId>) -> bool {
        let page = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry
                    .manifest
                    .required_resources()
                    .iter()
                    .any(|resource| !pinned.contains(&resource.id))
            })
            .min_by_key(|(_, entry)| entry.used)
            .map(|(page, _)| *page);
        page.is_some_and(|page| self.entries.remove(&page).is_some())
    }
}

/// Releases least-recently-used resources while the canvas exceeds its budget. Resources in
/// `pinned` (the active and staged frames) stay resident. Cached pages keep theirs until the
/// budget forces the least recently used page out; switching back to it prepares it again.
pub(crate) fn evict(
    resident: &mut impl ResidentResources,
    usage: &mut ResourceUsage,
    manifests: &mut PageManifests,
    pinned: &HashSet<ResourceId>,
) {
    let mut protected = pinned.clone();
    protected.extend(manifests.resources());
    while resident.resident_bytes() > RESOURCE_CACHE_BUDGET || usage.len() > MAX_CACHED_RESOURCES {
        if let Some(candidate) = usage.oldest_unprotected(&protected) {
            resident.release(candidate);
            usage.remove(candidate);
        } else if manifests.drop_least_recent(pinned) {
            protected.clone_from(pinned);
            protected.extend(manifests.resources());
        } else {
            break;
        }
    }
}

/// Caches `manifest` for `page`, then evicts over budget. Returns whether the page is still
/// cached: eviction drops it again when it does not fit beside the pinned frames.
pub(crate) fn cache_page(
    resident: &mut impl ResidentResources,
    usage: &mut ResourceUsage,
    manifests: &mut PageManifests,
    pinned: &HashSet<ResourceId>,
    page: LayerId,
    manifest: PreparedFrameManifest,
) -> bool {
    manifests.insert(page, manifest);
    evict(resident, usage, manifests, pinned);
    manifests.get(&page).is_some()
}

#[derive(Default)]
pub(crate) struct ResourceUsage {
    entries: HashMap<ResourceId, u64>,
    clock: u64,
}

impl ResourceUsage {
    pub(crate) fn insert(&mut self, resource: ResourceId) {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.entries.insert(resource, self.clock);
    }

    pub(crate) fn touch(&mut self, resource: ResourceId) {
        if self.entries.contains_key(&resource) {
            self.insert(resource);
        }
    }

    pub(crate) fn remove(&mut self, resource: ResourceId) {
        self.entries.remove(&resource);
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn oldest_unprotected(&self, protected: &HashSet<ResourceId>) -> Option<ResourceId> {
        self.entries
            .iter()
            .filter(|(id, _)| !protected.contains(id))
            .min_by_key(|(id, used)| (**used, **id))
            .map(|(id, _)| *id)
    }
}

#[cfg(test)]
mod tests {
    use koharu_rasterizer::{PreparedFrame, PreparedResourceKind, PreparedResourceRef, Revision};

    use super::*;

    /// One 4096x4096 RGBA raster, the size of a decoded scan on the GPU.
    const PAGE_BYTES: u64 = 64 * 1024 * 1024;

    fn id(value: u8) -> ResourceId {
        ResourceId::from_bytes([value; 32])
    }

    fn page(value: u8) -> LayerId {
        LayerId::from_bytes([value; 16])
    }

    fn manifest(value: u8) -> PreparedFrameManifest {
        PreparedFrameManifest {
            frame: PreparedFrame {
                revision: Revision::new(1),
                page: page(value),
                width: 4096,
                height: 4096,
                origin: (0, 0),
                normalization: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                layers: Vec::new(),
            },
            resources: vec![PreparedResourceRef {
                id: id(value),
                kind: PreparedResourceKind::EncodedRaster {
                    width: 4096,
                    height: 4096,
                },
                byte_len: PAGE_BYTES,
            }],
        }
    }

    impl ResidentResources for HashMap<ResourceId, u64> {
        fn resident_bytes(&self) -> u64 {
            self.values().sum()
        }

        fn release(&mut self, resource: ResourceId) {
            self.remove(&resource);
        }
    }

    #[derive(Default)]
    struct Canvas {
        resident: HashMap<ResourceId, u64>,
        usage: ResourceUsage,
        pages: PageManifests,
    }

    impl Canvas {
        /// Caches page `value` while page `active` is on screen, as activation and prefetch do.
        fn cache(&mut self, value: u8, active: u8) -> bool {
            self.resident.insert(id(value), PAGE_BYTES);
            self.usage.insert(id(value));
            cache_page(
                &mut self.resident,
                &mut self.usage,
                &mut self.pages,
                &HashSet::from([id(active)]),
                page(value),
                manifest(value),
            )
        }

        fn cached_pages(&self) -> Vec<u8> {
            (1..=u8::MAX)
                .filter(|value| self.pages.get(&page(*value)).is_some())
                .collect()
        }
    }

    #[test]
    fn browsing_past_the_budget_drops_the_least_recently_shown_pages() {
        let mut canvas = Canvas::default();
        for value in 1..=8 {
            canvas.cache(value, value);
        }
        canvas.cache(1, 1);
        for value in 9..=12 {
            canvas.cache(value, value);
        }

        assert_eq!(canvas.cached_pages(), [1, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(canvas.resident.resident_bytes(), RESOURCE_CACHE_BUDGET);
    }

    #[test]
    fn prefetched_pages_never_drop_the_active_page() {
        let mut canvas = Canvas::default();
        canvas.cache(1, 1);
        for value in 2..=10 {
            canvas.cache(value, 1);
        }

        assert_eq!(canvas.cached_pages(), [1, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn caching_reports_a_page_that_does_not_fit_beside_the_active_page() {
        let mut canvas = Canvas::default();
        canvas.cache(1, 1);
        // The active page alone fills the budget, so a prefetched page cannot stay beside it.
        canvas.resident.insert(id(1), RESOURCE_CACHE_BUDGET);

        assert!(
            !canvas.cache(2, 1),
            "the dropped page must not be reported as cached"
        );
        assert_eq!(canvas.cached_pages(), [1]);
        assert!(!canvas.resident.contains_key(&id(2)));
    }

    #[test]
    fn eviction_uses_least_recently_used_unprotected_resource() {
        let mut usage = ResourceUsage::default();
        usage.insert(id(1));
        usage.insert(id(2));
        usage.insert(id(3));
        usage.touch(id(1));

        assert_eq!(usage.len(), 3);
        assert_eq!(usage.oldest_unprotected(&HashSet::new()), Some(id(2)));
        usage.remove(id(2));
        assert_eq!(usage.oldest_unprotected(&HashSet::new()), Some(id(3)));
    }

    #[test]
    fn active_and_staged_resources_remain_pinned() {
        let mut usage = ResourceUsage::default();
        usage.insert(id(1));
        usage.insert(id(2));
        let protected = HashSet::from([id(1), id(2)]);

        assert_eq!(usage.oldest_unprotected(&protected), None);
    }
}
