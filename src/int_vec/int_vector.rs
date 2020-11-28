use std::fmt;

use super::*;
use bit_vec::{BitVec, BitVecMut};
use internal::vector_base::{self, VectorBase};
use space_usage::SpaceUsage;
use storage::BlockType;

/// Uncompressed vector of *k*-bit unsigned integers.
///
/// The element width *k* is determined at vector creation time.
///
/// `Block` gives the representation type. The element width *k* can
/// never exceed the number of bits in `Block`.
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct IntVector<Block: BlockType = usize> {
    element_bits: usize,
    base: VectorBase<Block>,
}

impl<Block: BlockType> IntVector<Block> {
    /// Asserts that `element_bits` is valid.
    fn check_element_bits(element_bits: usize) {
        assert!(
            element_bits != 0,
            "IntVector: cannot have zero-size elements"
        );
        assert!(
            element_bits <= Block::nbits(),
            "IntVector: element size cannot exceed block size"
        );
    }

    fn check_value_random(element_bits: usize, element_value: Block) {
        assert!(
            element_value <= Block::low_mask(element_bits),
            "IntVector: value to large for element size"
        );
    }

    fn check_value(&self, element_value: Block) {
        Self::check_value_random(self.element_bits, element_value);
    }

    /// Checks `element_bits` before assembling an `IntVector`.
    fn create(element_bits: usize, base: VectorBase<Block>) -> Self {
        Self::check_element_bits(element_bits);
        IntVector {
            element_bits: element_bits,
            base: base,
        }
    }

    #[inline]
    fn compute_address_random(
        &self,
        bit_offset: u64,
        element_bits: usize,
        element_index: u64,
    ) -> u64 {
        (element_index * element_bits as u64) + bit_offset
    }

    #[inline]
    fn compute_address(&self, element_index: u64) -> u64 {
        element_index * self.element_bits as u64
    }

    /// Creates a new integer vector.
    ///
    /// # Arguments
    ///
    ///  - `element_bits` — the size of each element in bits; hence
    ///    elements range from `0` to `2.pow(element_bits) - 1`.
    ///
    /// # Result
    ///
    /// The new, empty integer vector.
    #[inline]
    pub fn new(element_bits: usize) -> Self {
        Self::create(element_bits, VectorBase::new())
    }

    /// Creates a new, empty integer vector, allocating sufficient storage
    /// for `capacity` elements.
    #[inline]
    pub fn with_capacity(element_bits: usize, capacity: u64) -> Self {
        Self::create(
            element_bits,
            VectorBase::with_capacity(element_bits, capacity),
        )
    }

    /// Creates a new, empty integer vector, allocating `block_capacity`
    /// blocks of storage.
    #[inline]
    pub fn block_with_capacity(element_bits: usize, block_capacity: usize) -> Self {
        Self::create(
            element_bits,
            VectorBase::block_with_capacity(block_capacity),
        )
    }

    /// Creates a new integer vector containing `len` copies of `value`.
    #[inline]
    pub fn with_fill(element_bits: usize, len: u64, value: Block) -> Self {
        Self::create(
            element_bits,
            VectorBase::with_fill(element_bits, len, value),
        )
    }

    /// Creates a new integer vector containing `block_len` copies of the
    /// block `value`.
    ///
    /// The length of the new vector will be the number of elements of size
    /// `element_bits` that fit in `block_len` blocks.
    #[inline]
    pub fn block_with_fill(element_bits: usize, block_len: usize, value: Block) -> Self {
        Self::create(
            element_bits,
            VectorBase::block_with_fill(element_bits, block_len, value),
        )
    }

    /// Returns the element at a given index, also given an arbitrary
    /// element size and bit offset.
    ///
    /// This computes the location of the `element_index`th element
    /// supposing that elements are `element_bits` side, then adds
    /// `bit_offset` additional bits and returns the `element_bits`-bit
    /// value found at that location.
    ///
    /// # Panics
    ///
    /// Panics if the referenced bits are out of bounds. Bounds are
    /// considered to the end of the support array, even if that goes
    /// past the last element of the `IntArray`.
    #[inline]
    pub fn get_random(&self, bit_offset: u64, element_bits: usize, element_index: u64) -> Block {
        let address = self.compute_address_random(bit_offset, element_bits, element_index);
        self.base.get_bits(self.element_bits, address, element_bits)
    }

    /// Sets the element at a given index to a given value, also given
    /// an arbitrary element size and bit offset.
    ///
    /// This computes the location of the `element_index`th element
    /// supposing that elements are `element_bits` side, then adds
    /// `bit_offset` additional bits and modifies the `element_bits`-bit
    /// value found at that location.
    ///
    /// # Panics
    ///
    ///   - Panics if the referenced bits are out of bounds. Bounds are
    ///     considered to the end of the support array, even if that goes
    ///     past the last element of the `IntArray`.
    ///
    ///   - Debug mode only: Panics if `element_value` is too large to
    ///     fit in the element size. (TODO: What’s the right thing here?)
    #[inline]
    pub fn set_random(
        &mut self,
        bit_offset: u64,
        element_bits: usize,
        element_index: u64,
        element_value: Block,
    ) {
        assert!(element_value <= Block::low_mask(element_bits));

        let address = self.compute_address_random(bit_offset, element_bits, element_index);
        self.base
            .set_bits(self.element_bits, address, element_bits, element_value);
    }

    /// Pushes an element onto the end of the vector, increasing the
    /// length by 1.
    #[inline]
    pub fn push(&mut self, element_value: Block) {
        assert!(element_value <= Block::low_mask(self.element_bits));
        self.check_value(element_value);
        self.base.push_bits(self.element_bits, element_value);
    }

    /// Removes and returns the last element of the vector, if present.
    #[inline]
    pub fn pop(&mut self) -> Option<Block> {
        self.base.pop_bits(self.element_bits)
    }

    /// The number of elements the vector can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> u64 {
        self.base.capacity(self.element_bits)
    }

    /// The number of blocks of elements the vector can hold without
    /// reallocating.
    #[inline]
    pub fn block_capacity(&self) -> usize {
        self.base.block_capacity()
    }

    /// Resizes to the given number of elements, filling if necessary.
    #[inline]
    pub fn resize(&mut self, n_elements: u64, fill: Block) {
        self.base.resize(self.element_bits, n_elements, fill);
    }

    /// Resizes to the given number of blocks, filling if necessary.
    #[inline]
    pub fn block_resize(&mut self, n_blocks: usize, fill: Block) {
        self.base.block_resize(self.element_bits, n_blocks, fill);
    }

    /// Reserves capacity for at least `additional` more elements to be
    /// inserted in the given `IntVector<Block>`.
    ///
    /// The collection may reserve more space to avoid frequent
    /// reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the size conditions of
    /// [`IntVector::<Block>::is_okay_size()`](struct.IntVector.html#method.is_okay_size)
    /// are not met. This will happen if the total number of bits
    /// overflows `u64`.
    #[inline]
    pub fn reserve(&mut self, additional: u64) {
        self.base.reserve(self.element_bits, additional);
    }

    /// Reserves capacity for at least `additional` blocks of values to be
    /// inserted.
    ///
    /// The collection may reserve more space to avoid frequent
    /// reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the number of blocks overflows a `usize`.
    #[inline]
    pub fn block_reserve(&mut self, additional: usize) {
        self.base.block_reserve(additional);
    }

    /// Reserves capacity for at least `additional` more elements to be
    /// inserted in the given `IntVector<Block>`.
    ///
    /// Unlike [`reserve`](#method.reserve), does nothing if the
    /// capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the size conditions of
    /// [`IntVector::<Block>::is_okay_size()`](struct.IntVector.html#method.is_okay_size)
    /// are not met. This will happen if the total number of bits
    /// overflows `u64`.
    #[inline]
    pub fn reserve_exact(&mut self, additional: u64) {
        self.base.reserve_exact(self.element_bits, additional);
    }

    /// Reserves capacity for at least `additional` blocks of values to be
    /// inserted.
    ///
    /// Unlike [`reserve_block`](#method.reserve_block), does nothing if the
    /// capacity is already sufficient.
    ///
    /// The collection may reserve more space to avoid frequent
    /// reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the number of blocks overflows a `usize`.
    #[inline]
    pub fn block_reserve_exact(&mut self, additional: usize) {
        self.base.block_reserve_exact(additional);
    }

    /// Shrinks the capacity to just fit the number of elements.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.base.shrink_to_fit();
    }

    /// Shrinks to the given size.
    ///
    /// Does nothing if `n_elements` is greater than the current size.
    #[inline]
    pub fn truncate(&mut self, n_elements: u64) {
        self.base.truncate(self.element_bits, n_elements);
    }

    /// Shrinks to the given number of blocks.
    ///
    /// Does nothing if `n_blocks` is greater than the current blocks.
    #[inline]
    pub fn block_truncate(&mut self, n_blocks: usize) {
        self.base.block_truncate(self.element_bits, n_blocks);
    }

    /// Sets the size to 0 while retaining the allocated storage.
    #[inline]
    pub fn clear(&mut self) {
        self.base.clear();
    }

    /// Gets an iterator over the elements of the vector.
    #[inline]
    pub fn iter(&self) -> Iter<Block> {
        Iter(vector_base::Iter::new(self.element_bits, &self.base))
    }

    /// True if the element size matches the block size.
    #[inline]
    pub fn is_block_sized(&self) -> bool {
        self.element_bits() == Block::nbits()
    }

    /// True if elements are aligned within blocks.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        Block::nbits() % self.element_bits() == 0
    }
}

impl<Block: BlockType> IntVec for IntVector<Block> {
    type Block = Block;

    #[inline]
    fn len(&self) -> u64 {
        self.base.len()
    }

    #[inline]
    fn get(&self, element_index: u64) -> Block {
        if self.is_block_sized() {
            return self.base.get_block(element_index as usize);
        }

        let address = self.compute_address(element_index);
        self.base
            .get_bits(self.element_bits, address, self.element_bits)
    }

    #[inline]
    fn element_bits(&self) -> usize {
        self.element_bits
    }
}

impl<Block: BlockType> IntVecMut for IntVector<Block> {
    #[inline]
    fn set(&mut self, element_index: u64, element_value: Block) {
        if self.is_block_sized() {
            self.base
                .set_block(self.element_bits, element_index as usize, element_value);
            return;
        }

        // self.check_value(element_value);

        let address = self.compute_address(element_index);
        self.base
            .set_bits(self.element_bits, address, self.element_bits, element_value);
    }
}

impl<Block: BlockType> BitVec for IntVector<Block> {
    type Block = Block;

    #[inline]
    fn block_len(&self) -> usize {
        self.base.block_len()
    }

    #[inline]
    fn bit_len(&self) -> u64 {
        self.element_bits as u64 * self.base.len()
    }

    #[inline]
    fn get_block(&self, position: usize) -> Block {
        self.base.get_block(position)
    }
}

impl<Block: BlockType> BitVecMut for IntVector<Block> {
    #[inline]
    fn set_block(&mut self, position: usize, value: Block) {
        self.base.set_block(self.element_bits, position, value);
    }
}

/// An iterator over the elements of an [`IntVector`](struct.IntVector.html).
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Iter<'a, Block: BlockType + 'a = usize>(vector_base::Iter<'a, Block>);

impl<'a, Block: BlockType> Iterator for Iter<'a, Block> {
    type Item = Block;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.0.last()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n)
    }
}

#[cfg(target_pointer_width = "64")]
impl<'a, Block: BlockType> ExactSizeIterator for Iter<'a, Block> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, Block: BlockType> DoubleEndedIterator for Iter<'a, Block> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<'a, Block: BlockType + 'a> IntoIterator for &'a IntVector<Block> {
    type Item = Block;
    type IntoIter = Iter<'a, Block>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<Block> fmt::Debug for IntVector<Block>
where
    Block: BlockType + fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "IntVector {{ element_bits: {}, elements: {{ ",
            self.element_bits()
        )?;

        for element in self {
            write!(formatter, "{:?}, ", element)?;
        }

        write!(formatter, "}} }}")
    }
}

impl<A: BlockType> SpaceUsage for IntVector<A> {
    #[inline]
    fn is_stack_only() -> bool {
        false
    }

    #[inline]
    fn heap_bytes(&self) -> usize {
        self.base.heap_bytes()
    }
}

#[cfg(test)]
mod test {
    use bit_vec::*;
    use int_vec::{IntVec, IntVecMut, IntVector};

    #[test]
    fn create_empty() {
        let v: IntVector = IntVector::new(4);
        assert!(v.is_empty());
    }

    #[test]
    fn block_sized() {
        let mut v = IntVector::<u32>::with_fill(32, 10, 0);
        assert_eq!(10, v.len());

        assert_eq!(0, v.get(0));
        assert_eq!(0, v.get(9));

        v.set(0, 89);
        assert_eq!(89, v.get(0));
        assert_eq!(0, v.get(1));

        v.set(0, 56);
        v.set(1, 34);
        assert_eq!(56, v.get(0));
        assert_eq!(34, v.get(1));
        assert_eq!(0, v.get(2));

        v.set(9, 12);
        assert_eq!(12, v.get(9));
    }

    #[test]
    #[should_panic]
    fn block_sized_oob() {
        let v = IntVector::<u32>::with_fill(32, 10, 0);
        assert_eq!(0, v.get(10));
    }

    #[test]
    fn aligned() {
        let mut v = IntVector::<u32>::with_fill(4, 20, 0);
        assert_eq!(20, v.len());

        assert_eq!(0, v.get(0));
        assert_eq!(0, v.get(9));

        v.set(0, 13);
        assert_eq!(13, v.get(0));
        assert_eq!(0, v.get(1));

        v.set(1, 15);
        assert_eq!(13, v.get(0));
        assert_eq!(15, v.get(1));
        assert_eq!(0, v.get(2));

        v.set(1, 4);
        v.set(19, 9);
        assert_eq!(13, v.get(0));
        assert_eq!(4, v.get(1));
        assert_eq!(0, v.get(2));
        assert_eq!(9, v.get(19));
    }

    #[test]
    #[should_panic]
    fn aligned_oob() {
        let v = IntVector::<u32>::with_fill(4, 20, 0);
        assert_eq!(0, v.get(20));
    }

    #[test]
    fn unaligned() {
        let mut v = IntVector::<u32>::with_fill(5, 20, 0);
        assert_eq!(20, v.len());

        assert_eq!(0, v.get(0));
        assert_eq!(0, v.get(9));

        v.set(0, 13);
        assert_eq!(13, v.get(0));
        assert_eq!(0, v.get(1));

        v.set(1, 15);
        assert_eq!(13, v.get(0));
        assert_eq!(15, v.get(1));
        assert_eq!(0, v.get(2));

        v.set(1, 4);
        v.set(19, 9);
        assert_eq!(13, v.get(0));
        assert_eq!(4, v.get(1));
        assert_eq!(0, v.get(2));
        assert_eq!(9, v.get(19));
    }

    #[test]
    #[should_panic]
    fn unaligned_oob() {
        let v = IntVector::<u32>::with_fill(5, 20, 0);
        assert_eq!(0, v.get(20));
    }

    #[test]
    fn pop() {
        let mut v = IntVector::<u32>::new(7);
        assert_eq!(None, v.pop());
        v.push(1);
        v.push(2);
        v.push(3);
        assert_eq!(Some(3), v.pop());
        v.push(4);
        v.push(5);
        assert_eq!(Some(5), v.pop());
        assert_eq!(Some(4), v.pop());
        assert_eq!(Some(2), v.pop());
        assert_eq!(Some(1), v.pop());
        assert_eq!(None, v.pop());
    }

    #[test]
    fn iter() {
        let mut v = IntVector::<u16>::new(13);
        v.push(1);
        v.push(1);
        v.push(2);
        v.push(3);
        v.push(5);

        assert_eq!(vec![1, 1, 2, 3, 5], v.iter().collect::<Vec<_>>());
    }

    #[test]
    fn debug() {
        let mut v = IntVector::<u16>::new(13);
        v.push(1);
        v.push(1);
        v.push(2);
        v.push(3);
        v.push(5);

        assert_eq!(
            "IntVector { element_bits: 13, elements: { 1, 1, 2, 3, 5, } }".to_owned(),
            format!("{:?}", v)
        );
    }

    #[test]
    #[should_panic]
    fn value_overflow() {
        let mut v = IntVector::<u32>::new(3);
        v.push(78); // 78 is too big
    }

    #[test]
    fn bit_vec() {
        let mut v = IntVector::<u32>::new(1);
        v.push(1);
        v.push(0);
        v.push(0);
        v.push(1);

        assert!(v.get_bit(0));
        assert!(!v.get_bit(1));
        assert!(!v.get_bit(2));
        assert!(v.get_bit(3));

        v.set_bit(1, true);

        assert!(v.get_bit(0));
        assert!(v.get_bit(1));
        assert!(!v.get_bit(2));
        assert!(v.get_bit(3));
    }

    #[test]
    fn push_pop_equals() {
        let mut v = IntVector::<u32>::new(5);
        let mut u = IntVector::<u32>::new(5);

        v.push(5);
        u.push(5);
        assert!(v == u);

        v.push(6);
        u.push(7);
        assert!(v != u);

        v.pop();
        u.pop();
        assert!(v == u);
    }

    #[test]
    fn block_size_elements_u16() {
        let mut v = IntVector::<u16>::new(16);
        v.push(0);
        v.push(!0);
        assert_eq!(Some(!0), v.pop());
        assert_eq!(Some(0), v.pop());
        assert_eq!(None, v.pop());
    }

    #[test]
    fn block_size_elements_u64() {
        let mut v = IntVector::<u64>::new(64);
        v.push(0);
        v.push(!0);
        assert_eq!(Some(!0), v.pop());
        assert_eq!(Some(0), v.pop());
        assert_eq!(None, v.pop());
    }
}
