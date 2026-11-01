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
    entries: HashMap<LayerId, PreparedFrameManifest>,
}

impl PageManifests {
    pub(crate) fn get(&self, page: &LayerId) -> Option<&PreparedFrameManifest> {
        self.entries.get(page)
    }

    pub(crate) fn insert(&mut self, page: LayerId, manifest: PreparedFrameManifest) {
        self.entries.insert(page, manifest);
    }

    fn resources(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.entries.values().flat_map(|manifest| {
            manifest
                .required_resources()
                .iter()
                .map(|resource| resource.id)
        })
    }
}

/// Releases least-recently-used resources while the canvas exceeds its budget. Resources in
/// `pinned` (the active and staged frames) and those of cached pages stay resident.
pub(crate) fn evict(
    resident: &mut impl ResidentResources,
    usage: &mut ResourceUsage,
    manifests: &mut PageManifests,
    pinned: &HashSet<ResourceId>,
) {
    let mut protected = pinned.clone();
    protected.extend(manifests.resources());
    while resident.resident_bytes() > RESOURCE_CACHE_BUDGET || usage.len() > MAX_CACHED_RESOURCES {
        let Some(candidate) = usage.oldest_unprotected(&protected) else {
            break;
        };
        resident.release(candidate);
        usage.remove(candidate);
    }
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
    use super::*;

    fn id(value: u8) -> ResourceId {
        ResourceId::from_bytes([value; 32])
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
