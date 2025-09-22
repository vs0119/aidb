#[derive(Clone, Debug)]
pub enum CompactionStrategy {
    SizeTiered {
        level0_trigger: usize,
        fanout: usize,
    },
    Leveled {
        level_base: usize,
        level_multiplier: usize,
    },
}

impl CompactionStrategy {
    pub fn max_tables_for_level(&self, level: usize) -> usize {
        match self {
            CompactionStrategy::SizeTiered {
                level0_trigger,
                fanout,
            } => {
                if level == 0 {
                    *level0_trigger
                } else {
                    (*level0_trigger).saturating_mul(pow(*fanout, level))
                }
            }
            CompactionStrategy::Leveled {
                level_base,
                level_multiplier,
            } => {
                let multiplier = if level == 0 {
                    1
                } else {
                    pow(*level_multiplier, level)
                };
                (*level_base).saturating_mul(multiplier.max(1))
            }
        }
    }

    pub fn should_compact(&self, level: usize, level_sizes: &[usize]) -> bool {
        if let Some(&size) = level_sizes.get(level) {
            size > self.max_tables_for_level(level)
        } else {
            false
        }
    }

    pub fn tables_to_compact(&self, level: usize, level_sizes: &[usize]) -> usize {
        if let Some(&size) = level_sizes.get(level) {
            match self {
                CompactionStrategy::SizeTiered { fanout, .. } => size.min((*fanout).max(1)),
                CompactionStrategy::Leveled { level_base, .. } => {
                    size.saturating_sub((*level_base).max(1)).max(1)
                }
            }
        } else {
            0
        }
    }
}

impl Default for CompactionStrategy {
    fn default() -> Self {
        CompactionStrategy::SizeTiered {
            level0_trigger: 4,
            fanout: 2,
        }
    }
}

fn pow(base: usize, exp: usize) -> usize {
    if exp == 0 {
        return 1;
    }
    let mut result = 1usize;
    for _ in 0..exp {
        result = result.saturating_mul(base.max(1));
    }
    result
}
