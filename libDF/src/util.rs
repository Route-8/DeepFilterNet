use std::cell::{RefCell, UnsafeCell};
use std::convert::Infallible;
use std::rc::Rc;
use std::thread_local;

use rand::distr::{
    uniform::{SampleUniform, Uniform},
    Distribution,
};
use rand::{RngExt, TryRng};
use rand_xoshiro::rand_core::{Rng as XoshiroRng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use thiserror::Error;

#[cfg(feature = "logging")]
pub use crate::logging::*;

type Result<T> = std::result::Result<T, UtilsError>;

#[derive(Error, Debug)]
pub enum UtilsError {
    #[error("Random seed is not initialized using seed_from_u64(x)")]
    SeedNotInitialized,
    #[error("Could not inititalize logger")]
    SetLoggerError(#[from] log::SetLoggerError),
}

pub struct SeededRng {
    rng: Rc<UnsafeCell<Xoshiro256PlusPlus>>,
}
thread_local!(
    static THREAD_SEEDED_RNG: Rc<UnsafeCell<Xoshiro256PlusPlus>> =
        Rc::new(UnsafeCell::new(Xoshiro256PlusPlus::seed_from_u64(0)));
    static SEEDED: RefCell<bool> = const { RefCell::new(false) };
);

pub fn seed_from_u64(x: u64) {
    SEEDED.with(|s| s.replace(true));
    unsafe { THREAD_SEEDED_RNG.with(|rng| *rng.get() = Xoshiro256PlusPlus::seed_from_u64(x)) }
}

impl TryRng for SeededRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> std::result::Result<u32, Self::Error> {
        Ok(unsafe { (*self.rng.get()).next_u32() })
    }
    fn try_next_u64(&mut self) -> std::result::Result<u64, Self::Error> {
        Ok(unsafe { (*self.rng.get()).next_u64() })
    }
    fn try_fill_bytes(&mut self, slice: &mut [u8]) -> std::result::Result<(), Self::Error> {
        unsafe { (*self.rng.get()).fill_bytes(slice) };
        Ok(())
    }
}

pub fn thread_rng() -> Result<SeededRng> {
    if !(SEEDED.with(|s| *s.borrow())) {
        return Err(UtilsError::SeedNotInitialized);
    }
    Ok(SeededRng {
        rng: THREAD_SEEDED_RNG.with(|rng| rng.clone()),
    })
}

impl SeededRng {
    #[inline]
    pub fn log_uniform(&mut self, low: f32, high: f32) -> f32 {
        self.random_range(low.ln()..=high.ln()).exp()
    }
    #[inline]
    pub fn uniform<T: SampleUniform + PartialOrd>(&mut self, low: T, high: T) -> T {
        if low >= high {
            low
        } else {
            self.random_range(low..high)
        }
    }
    #[inline]
    pub fn uniform_inclusive<T: SampleUniform + PartialOrd>(&mut self, low: T, high: T) -> T {
        if low >= high {
            low
        } else {
            self.random_range(low..=high)
        }
    }
}

pub(crate) fn rng_uniform<T>(n: usize, low: T, high: T) -> Result<Vec<T>>
where
    T: Default + Clone + SampleUniform,
{
    let mut rng = thread_rng()?;
    let mut v = vec![T::default(); n];
    let dist = Uniform::new_inclusive(low, high).unwrap();
    for x in v.iter_mut() {
        *x = dist.sample(&mut rng);
    }
    Ok(v)
}
