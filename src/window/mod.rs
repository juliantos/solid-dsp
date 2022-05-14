extern crate alloc;

use alloc::alloc::Layout;
use std::{mem, ptr};

use std::fmt;

#[derive(Debug)]
pub struct Window<T> {
    layout: Layout,
    delay: usize,
    capacity: usize,
    buffer: *mut T,
}

impl<T: Copy> Window<T> {
    pub fn new(capacity: usize, delay: usize) -> Self {
        assert!(capacity > 0);
        let alignment = mem::align_of::<T>();
        let size = mem::size_of::<T>();
        let layout = match Layout::from_size_align(size * (capacity + delay), alignment) {
            Ok(layout) => layout,
            _ => panic!("Unable to create Window of {}", capacity + delay)
        };
        let ptr = unsafe { alloc::alloc::alloc_zeroed(layout) } as *mut T;

        Window {
            layout: layout,
            delay: delay,
            capacity: capacity,
            buffer: ptr,
        }
    }

    pub fn as_ptr(&self) -> *const T {
        unsafe {
            let ptr = alloc::alloc::alloc(self.layout) as *mut T;
            std::ptr::copy(self.buffer.offset(self.delay as isize), ptr, self.capacity);
            ptr
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        let mut destination = Vec::with_capacity(self.capacity);
        unsafe {
            ptr::copy(self.as_ptr(), destination.as_mut_ptr(), self.capacity);
            destination.set_len(self.capacity);
        }
        destination
    }

    #[inline]
    pub fn reset(&mut self) {
        self.buffer = unsafe { alloc::alloc::alloc_zeroed(self.layout) } as *mut T;
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn push(&mut self, element: T) {
        unsafe {
            std::ptr::copy(self.buffer,
                self.buffer.offset(1),
                self.capacity - 1);
        }

        unsafe {
            ptr::write(self.buffer, element);
        }
    }

    pub fn write(&mut self, other: &[T]) {
        for element in other.iter() {
            self.push(*element);
        }
    }
}

impl<T: Clone> Clone for Window<T> where T: Copy {
    fn clone(&self) -> Self {
        let svec = self.to_vec();
        let mut new_window_buffer = Window::<T>::new(self.capacity, self.delay);
        new_window_buffer.write(&svec);
        new_window_buffer
    }
}

impl<T: fmt::Display + std::string::ToString> fmt::Display for Window<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut values: String = "".to_string();
        for i in 0..self.capacity {
            let value = unsafe {
                let buf_ptr = self.buffer.offset((self.delay + i) as isize);
                ptr::read(buf_ptr)
            };
            values += &value.to_string();
            if i != self.capacity - 1 {
                values += ", "
            }
        }
        write!(f, "[{}]", values)
    }
}