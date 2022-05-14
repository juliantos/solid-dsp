//! A non-contigous static array type with heap allocated contents, written [`CircularBuffer<T>`].
//! 
//! Circular Buffers have `O(1)` push (to the end) and `O(1)` pop (from the front)
//! 
//! Circular Buffers never allocated more than `isize::MAX` bytes;
//! 
//! # Example
//! 
//! You can explicitly create a [`CircularBuffer`] with [`CircularBuffer::new`]
//! 
//! ```
//! use solid::circular_buffer::CircularBuffer;
//! let buffer: CircularBuffer<u8> = CircularBuffer::new(10);
//! ```
extern crate alloc;

use alloc::alloc::Layout;
use std::{mem, ptr};

use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq)]
#[allow(dead_code)]
pub enum BufferErrorCode {
    EmptyBuffer,
    FullBuffer,
    NotEnoughBuffer,
    NegativeBuffer,
    NonExistantBuffer
}

#[derive(Debug)]
pub struct BufferError(pub BufferErrorCode);

impl fmt::Display for BufferError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let error_name = match self.0 {
            BufferErrorCode::EmptyBuffer => "Empty Buffer",
            BufferErrorCode::FullBuffer => "Full Buffer",
            BufferErrorCode::NotEnoughBuffer => "Not Enough Buffer",
            BufferErrorCode::NegativeBuffer => "Negative Buffer",
            BufferErrorCode::NonExistantBuffer => "Non Existant Buffer"
        };
        write!(f, r#"Buffer Error: {}"#, error_name)
    }
}

impl std::error::Error for BufferError {}

#[derive(Debug)]
#[allow(dead_code)]
pub struct CircularBuffer<T> {
    layout: Layout,
    capacity: isize,
    buffer: *mut T,
    read_index: isize,
    write_index: isize,
    num_elements: isize
}

#[allow(dead_code)]
impl<T> CircularBuffer<T> {
    /// Constructs a new, empty `CircularBuffer<T>` 
    /// 
    /// The Buffer Allocates before any element is pushed on it
    /// 
    /// Creates a Circular Buffer of the type `T` and allocates 
    /// n amount of elements. These values are all initialized to
    /// 0.
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(10);
    /// ```
    pub fn new(capacity: isize) -> Self {
        assert!(capacity > 0);
        let alignment = mem::align_of::<T>();
        let size = mem::size_of::<T>();
        let layout = match Layout::from_size_align(size * capacity as usize, alignment) {
            Ok(layout) => layout,
            Err(_) => panic!("Unable to create Circular Buffer of {}", capacity)
        };
        let ptr = unsafe { alloc::alloc::alloc(layout) } as *mut T;
        CircularBuffer{
            layout: layout,
            capacity: capacity as isize, 
            buffer: ptr,
            read_index: 0,
            write_index: 0,
            num_elements: 0
        }
    }

    /// Returns a raw pointer to the start of the buffer
    /// 
    /// This function also re-linearizes the data so that the data
    /// at the front of the function is in order and not wrapped
    /// 
    /// The caller also must make sure that the Circular Buffer outlives
    /// the pointer this function returns, otherwise it will point to garbage.
    /// Modifying the Circular Buffer also may invalidate the pointer.
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(10);
    /// let buffer_ptr = buffer.as_ptr();
    /// 
    /// unsafe {
    ///     for i in 0..buffer.len() {
    ///         assert_eq!(*buffer_ptr.add(i as usize), 0);
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        unsafe {
            let ptr = alloc::alloc::alloc(self.layout) as *mut T;
            std::ptr::copy(
                self.buffer.offset(self.read_index),
                ptr,
                (self.capacity - self.read_index) as usize);
            std::ptr::copy(
                self.buffer,
                ptr.offset(self.capacity - self.read_index),
                (self.read_index) as usize);
            ptr
        }
    }

    /// Returns a [`Vec<T>`] that of the ptr of the buffer
    /// 
    /// This method calls the [`CircularBuffer::as_ptr`] method and inherents all the attributes
    /// of that method.
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(10);
    /// let buffer_vec = buffer.to_vec();
    /// for i in 0..buffer_vec.len() {
    ///     assert_eq!(buffer_vec[i], 0);
    /// }
    /// ```
    #[inline]
    pub fn to_vec(&self) -> Vec<T> {
        let mut destination = Vec::with_capacity(self.capacity as usize);
        unsafe {
            ptr::copy(self.as_ptr(), destination.as_mut_ptr(), self.len() as usize);
            destination.set_len(self.len() as usize);
        }
        destination
    }

    /// Resets the read offset, the write offset and number of
    /// elements to 0. 
    /// 
    /// This function resets to a known place to setup the a 
    /// semi-clean slate without doing any memory manipulation.
    /// This allows for a quick and reliable circular buffer reset.
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(10);
    /// 
    /// buffer.push(1);
    /// assert_eq!(buffer.len(), 1);
    /// 
    /// buffer.reset();
    /// assert_eq!(buffer.len(), 0);
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.read_index = 0;
        self.write_index = 0;
        self.num_elements = 0;
    }

    /// Returns the number of elements in the Circular Buffer,
    /// also referred as its 'length'.
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(10);
    /// buffer.push(1);
    /// buffer.push(2);
    /// buffer.push(3);
    /// 
    /// assert_eq!(buffer.len(), 3);
    /// 
    /// buffer.pop();
    /// 
    /// assert_eq!(buffer.len(), 2);
    /// ```
    #[inline]
    pub fn len(&self) -> isize {
        self.num_elements
    }

    /// Returns the full capacity allocated for the Circular Buffer.
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(10);
    /// assert_eq!(buffer.capacity(), 10);
    /// ```
    #[inline]
    pub fn capacity(&self) -> isize {
        self.capacity
    }

    /// Returns the amount left that can be set without overflowing the
    /// Circular Buffer.
    /// It is also a quick hand for the `capacity` - `len`
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(10);
    /// buffer.push(1);
    /// 
    /// assert_eq!(buffer.reserved(), 9);
    /// ```
    #[inline]
    pub fn reserved(&self) -> isize {
        self.capacity - self.num_elements
    }

    /// Returns `true` if the Circular Buffer contains no elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(10);
    /// 
    /// assert_eq!(buffer.is_empty(), true);
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_elements == 0
    }

    /// Returns `true` if the Circular Buffer has as many elements
    /// as the set capacity.
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(3);
    /// buffer.push(1);
    /// buffer.push(2);
    /// buffer.push(3);
    /// 
    /// assert_eq!(buffer.is_full(), true);
    /// ```
    #[inline]
    pub fn is_full(&self) -> bool {
        self.num_elements == self.capacity
    }

    /// Returns the index from `0` to `capacity` that the read 
    /// index is pointing.
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(10);
    /// 
    /// buffer.push(1);
    /// buffer.push(2);
    /// 
    /// let val = buffer.pop().unwrap();
    /// 
    /// assert_eq!(buffer.read_index(), 1);
    /// ```
    #[inline]
    pub fn read_index(&self) -> isize {
        self.read_index
    }

    /// Returns the index from `0` to `capacity` that the next index
    /// to start writing at is.
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut buffer = solid::circular_buffer::CircularBuffer::<u8>::new(10);
    /// 
    /// assert_eq!(buffer.write_index(), 0);
    /// 
    /// buffer.push(1);
    /// 
    /// assert_eq!(buffer.write_index(), 1);
    /// ```
    #[inline]
    pub fn write_index(&self) -> isize {
        self.write_index
    }

    /// Tries to push an element onto the Circular Buffer.
    /// 
    /// If the buffer is full, it will fail and return a 
    /// [`BufferErrorCode::FullBuffer`]. This method also advances
    /// the write index by 1 and wraps if need be.
    /// 
    /// # Example
    /// ```
    /// use ::solid::circular_buffer::{CircularBuffer, BufferError, BufferErrorCode};
    /// let mut buffer = CircularBuffer::<u8>::new(1);
    /// 
    /// assert_eq!(buffer.push(1).unwrap(), ());
    /// 
    /// assert_eq!(buffer.push(2).unwrap_err().downcast_ref::<BufferError>().unwrap().0, BufferErrorCode::FullBuffer);
    /// ```
    pub fn push(&mut self, element: T) -> Result<(), Box<dyn Error>> {
        if self.is_full() { return Err(Box::new(BufferError(BufferErrorCode::FullBuffer))) }
        else {
            unsafe {
                let write_ptr = self.buffer.offset((self.write_index % self.capacity) as isize);
                ptr::write(write_ptr, element);
            }
            self.write_index = (self.write_index + 1) % self.capacity;
            self.num_elements += 1;
            Ok(())
        }
    }

    /// Tries to write a number of elements onto the Circular Buffer
    /// 
    /// If the buffer is full, it will fail and return a 
    /// [`BufferErrorCode::NotEnoughBuffer`]. This method also advances 
    /// the write index by `size` and wraps if need be.
    /// 
    /// # Example
    /// ```
    /// use ::solid::circular_buffer::{CircularBuffer, BufferError, BufferErrorCode};
    /// let mut buffer = CircularBuffer::<u8>::new(4);
    /// 
    /// assert_eq!(buffer.write(vec!(2,3,4,5).as_ptr(), 4).unwrap(), ());
    /// 
    /// assert_eq!(buffer.write(vec!(6).as_ptr(), 1).unwrap_err().downcast_ref::<BufferError>().unwrap().0, BufferErrorCode::NotEnoughBuffer);
    /// ```
    pub fn write(&mut self, other: *const T, size: usize) -> Result<(), Box<dyn Error>> {
        if self.num_elements + size as isize > self.capacity {
            return Err(Box::new(BufferError(BufferErrorCode::NotEnoughBuffer)))
        } else {
            unsafe {
                if size as isize <= self.capacity - self.write_index {
                    std::ptr::copy_nonoverlapping(other, self.buffer.offset(self.write_index), size);
                    self.write_index = (self.write_index + size as isize) % self.capacity;
                    self.num_elements += size as isize;
                    Ok(())
                } else {
                    std::ptr::copy_nonoverlapping(
                        other, 
                        self.buffer.offset(self.write_index), 
                        (self.capacity - self.write_index) as usize);
                    std::ptr::copy_nonoverlapping(
                        other.offset(size as isize - (self.capacity - self.write_index)), 
                        self.buffer, 
                        size - (self.capacity - self.write_index) as usize);
                    self.write_index = (self.write_index + size as isize) % self.capacity;
                    self.num_elements += size as isize;
                    Ok(())
                }
            }
        }
    }

    /// Tries to pop an element from the Circular Buffer.
    /// 
    /// If the buffer is empty, it will fail and return a 
    /// [`BufferErrorCode::EmptyBuffer`]. This method also advances
    /// the read index by 1 and wraps if need be.
    /// 
    /// # Example
    /// ```
    /// use ::solid::circular_buffer::{CircularBuffer, BufferError, BufferErrorCode};
    /// let mut buffer = CircularBuffer::<u8>::new(10);
    /// 
    /// assert_eq!(buffer.pop().unwrap_err().downcast_ref::<BufferError>().unwrap().0, BufferErrorCode::EmptyBuffer);
    /// 
    /// buffer.push(1);
    /// assert_eq!(buffer.pop().unwrap(), 1);
    /// ```
    pub fn pop(&mut self) -> Result<T, Box<dyn Error>> {
        if self.is_empty() { return Err(Box::new(BufferError(BufferErrorCode::EmptyBuffer))) }
        else {
            let value = unsafe {
                let read_ptr = self.buffer.offset(self.read_index);
                ptr::read(read_ptr)
            };
            self.read_index = (self.read_index + 1) % self.capacity;
            self.num_elements -= 1;
            Ok(value)
        }
    }

    /// This method moves the read pointer by `n` amount of elements.
    /// 
    /// If `n` happens to be negative then [`BufferErrorCode::NegativeBuffer`]
    /// will be returned. If `n` is greater than the amount of elements
    /// that can be read then [`BufferErrorCode::NotEnoughBuffer`] is returned.
    /// 
    /// This method is lightweight and faster than using [`CircularBuffer::pop`]
    /// since it is just advancing the read index by the amount.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::circular_buffer::{CircularBuffer, BufferError, BufferErrorCode};
    /// let mut buffer = CircularBuffer::<u8>::new(10);
    /// 
    /// assert_eq!(buffer.release(-1).unwrap_err().downcast_ref::<BufferError>().unwrap().0, BufferErrorCode::NegativeBuffer);
    /// assert_eq!(buffer.release(1).unwrap_err().downcast_ref::<BufferError>().unwrap().0, BufferErrorCode::NotEnoughBuffer);
    /// 
    /// buffer.push(1);
    /// assert_eq!(buffer.release(1).unwrap(), ());
    /// ```
    #[inline]
    pub fn release(&mut self, n: isize) -> Result<(), Box<dyn Error>> {
        if n < 0 {
            return Err(Box::new(BufferError(BufferErrorCode::NegativeBuffer)))
        } else if n > self.num_elements {
            return Err(Box::new(BufferError(BufferErrorCode::NotEnoughBuffer)))
        }
        self.read_index = (self.read_index + n) % self.capacity;
        self.num_elements -= n;
        Ok(())
    }
}

impl<T> Drop for CircularBuffer<T> {
    fn drop(&mut self) {
        match self.release(self.len()) {
            Ok(_) => (),
            Err(_) => panic!("Failed to release values from Circular Buffer!")
        };
        self.reset();
        unsafe { alloc::alloc::dealloc(self.buffer as *mut u8, self.layout) };
    }
}

impl<T: Clone> Clone for CircularBuffer<T> {
    fn clone(&self) -> Self {
        let ptr: *const T = self.as_ptr();
        let mut new_circular_buffer = CircularBuffer::<T>::new(self.capacity());
        match new_circular_buffer.write(ptr, self.num_elements as usize) {
            Ok(_) => return new_circular_buffer,
            Err(_) => panic!("Failed to Clone Circular Buffer!")
        };
    }
}

impl<T: fmt::Display + std::string::ToString> fmt::Display for CircularBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut values: String = "".to_string();
        for i in 0..self.len() {
            let value = unsafe {
                let buf_ptr = self.buffer.offset((self.read_index + i) % self.capacity);
                ptr::read(buf_ptr)
            };
            values += &value.to_string();
            if i != self.len() - 1 {
                values += ", "
            }
        }
        write!(f, "[{}]", values)
    }
}