#![no_std]

#[macro_export]
macro_rules! bitflags {
    (
        $(#[$outer:meta])*
        pub flags $BitFlags:ident: $T:ty {
            $(
                $(#[$inner:meta])*
                const $Flag:ident = $value:expr,
            )*
        }
    ) => {
        $(#[$outer])*
        #[derive(Copy, Clone, PartialEq, Eq, Hash)]
        pub struct $BitFlags {
            bits: $T,
        }

        impl $BitFlags {
            pub const fn empty() -> Self {
                Self { bits: 0 }
            }

            pub const fn bits(&self) -> $T {
                self.bits
            }

            pub const fn from_bits_truncate(bits: $T) -> Self {
                Self { bits }
            }
        }

        impl ::core::default::Default for $BitFlags {
            fn default() -> Self {
                Self::empty()
            }
        }

        impl ::core::fmt::Debug for $BitFlags {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                f.debug_struct(stringify!($BitFlags))
                    .field("bits", &self.bits)
                    .finish()
            }
        }

        impl ::core::ops::BitOr for $BitFlags {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self::Output {
                Self { bits: self.bits | rhs.bits }
            }
        }

        impl ::core::ops::BitOrAssign for $BitFlags {
            fn bitor_assign(&mut self, rhs: Self) {
                self.bits |= rhs.bits;
            }
        }

        impl ::core::ops::BitAnd for $BitFlags {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self::Output {
                Self { bits: self.bits & rhs.bits }
            }
        }

        impl ::core::ops::BitAndAssign for $BitFlags {
            fn bitand_assign(&mut self, rhs: Self) {
                self.bits &= rhs.bits;
            }
        }

        impl ::core::ops::Not for $BitFlags {
            type Output = Self;

            fn not(self) -> Self::Output {
                Self { bits: !self.bits }
            }
        }

        $(
            $(#[$inner])*
            pub const $Flag: $BitFlags = $BitFlags { bits: $value };
        )*
    };
}
