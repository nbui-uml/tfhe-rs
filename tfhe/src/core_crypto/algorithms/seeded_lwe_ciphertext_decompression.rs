//! Module with primitives pertaining to [`SeededLweCiphertext`] decompression.

use crate::core_crypto::algorithms::slice_algorithms::slice_wrapping_scalar_mul_assign;
use crate::core_crypto::commons::math::random::RandomGenerator;
use crate::core_crypto::commons::traits::*;
use crate::core_crypto::entities::*;

/// Convenience function to share the core logic of the decompression algorithm for
/// [`SeededLweCiphertext`] between all functions needing it.
pub fn decompress_seeded_lwe_ciphertext_with_existing_generator<Scalar, OutputCont, Gen>(
    output_lwe: &mut LweCiphertext<OutputCont>,
    input_seeded_lwe: &SeededLweCiphertext<Scalar>,
    generator: &mut RandomGenerator<Gen>,
) where
    Scalar: UnsignedTorus,
    OutputCont: ContainerMut<Element = Scalar>,
    Gen: ByteRandomGenerator,
{
    assert_eq!(
        output_lwe.ciphertext_modulus(),
        input_seeded_lwe.ciphertext_modulus(),
        "Mismatched CiphertextModulus \
    between input SeededLweCiphertext ({:?}) and output LweCiphertext ({:?})",
        input_seeded_lwe.ciphertext_modulus(),
        output_lwe.ciphertext_modulus(),
    );

    let ciphertext_modulus = output_lwe.ciphertext_modulus();
    let (mut output_mask, output_body) = output_lwe.get_mut_mask_and_body();

    // generate a uniformly random mask
    generator.fill_slice_with_random_uniform_custom_mod(output_mask.as_mut(), ciphertext_modulus);
    if !ciphertext_modulus.is_native_modulus() {
        slice_wrapping_scalar_mul_assign(
            output_mask.as_mut(),
            ciphertext_modulus.get_scaling_to_native_torus(),
        );
    }
    *output_body.data = *input_seeded_lwe.get_body().data;
}

/// Decompress a [`SeededLweCiphertext`], without consuming it, into a standard
/// [`LweCiphertext`].
pub fn decompress_seeded_lwe_ciphertext<Scalar, OutputCont, Gen>(
    output_lwe: &mut LweCiphertext<OutputCont>,
    input_seeded_lwe: &SeededLweCiphertext<Scalar>,
) where
    Scalar: UnsignedTorus,
    OutputCont: ContainerMut<Element = Scalar>,
    Gen: ByteRandomGenerator,
{
    assert_eq!(
        output_lwe.ciphertext_modulus(),
        input_seeded_lwe.ciphertext_modulus(),
        "Mismatched CiphertextModulus \
    between input SeededLweCiphertext ({:?}) and output LweCiphertext ({:?})",
        input_seeded_lwe.ciphertext_modulus(),
        output_lwe.ciphertext_modulus(),
    );

    let mut generator = RandomGenerator::<Gen>::new(input_seeded_lwe.compression_seed().seed);
    decompress_seeded_lwe_ciphertext_with_existing_generator::<_, _, Gen>(
        output_lwe,
        input_seeded_lwe,
        &mut generator,
    )
}
