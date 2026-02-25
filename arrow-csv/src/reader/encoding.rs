// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::fmt::Debug;
use std::io::{BufRead, BufReader, Read};

use encoding_rs::{CoderResult, Decoder, Encoding};

/// A decoder that converts a byte stream into UTF-8
pub struct CharsetDecoder {
    decoder: Decoder,
    eof: bool,
}

impl CharsetDecoder {
    /// Creates a new `CharsetDecoder` that decodes bytes in the
    /// specified encoding into UTF-8 bytes
    pub fn new(encoding: &'static Encoding) -> Self {
        Self {
            decoder: Encoding::new_decoder(encoding),
            eof: false,
        }
    }

    /// Decodes bytes in `input` and writes them to `output`,
    /// returning the number of bytes read and bytes written
    pub fn decode(&mut self, input: &[u8], output: &mut [u8], last: bool) -> (usize, usize) {
        if self.eof {
            return (0, 0);
        }

        let (result, read, written, _) = self.decoder.decode_to_utf8(input, output, last);

        if last && result == CoderResult::InputEmpty {
            self.eof = true;
        }

        (read, written)
    }

    /// Returns `true` if the decoder is finished and all bytes have been written out
    pub fn is_eof(&self) -> bool {
        self.eof
    }
}

// Manual implementation needed because `Decoder` doesn't implement Debug
impl Debug for CharsetDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CharsetDecoder")
            .field("decoder", self.decoder.encoding())
            .field("eof", &self.eof)
            .finish()
    }
}

/// A wrapper around a reader that optionally converts from a
/// specified character encoding to UTF-8 bytes
pub struct CharsetDecoderReader<R> {
    reader: R,
    decoder: Option<CharsetDecoder>,
}

impl<R: Read> CharsetDecoderReader<BufReader<R>> {
    pub fn new(reader: R, encoding: Option<&'static Encoding>) -> Self {
        Self {
            reader: BufReader::new(reader),
            decoder: encoding.map(CharsetDecoder::new),
        }
    }
}

impl<R: BufRead> Read for CharsetDecoderReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let Some(decoder) = self.decoder.as_mut() else {
            return self.reader.read(buf);
        };

        let src = self.reader.fill_buf()?;
        let (read, written) = decoder.decode(src, buf, src.is_empty());

        self.reader.consume(read);

        Ok(written)
    }
}

pub mod buffer {
    /// A fixed-sized buffer that maintains both
    /// a read position and a write position
    #[derive(Debug)]
    pub struct Buffer {
        buf: Box<[u8]>,
        read_ptr: usize,
        write_ptr: usize,
    }

    impl Buffer {
        /// Creates a new `Buffer` with the specified capacity
        #[inline]
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                buf: vec![0; capacity].into_boxed_slice(),
                read_ptr: 0,
                write_ptr: 0,
            }
        }

        /// Whether there are no more bytes available to be read
        pub fn is_empty(&self) -> bool {
            self.read_ptr == self.write_ptr
        }

        /// Returns the unread portion of the buffer
        pub fn read_buf(&self) -> &[u8] {
            &self.buf[self.read_ptr..self.write_ptr]
        }

        /// Advances the read position by `amount` bytes
        pub fn consume(&mut self, amount: usize) {
            self.read_ptr += amount;
            debug_assert!(self.read_ptr <= self.write_ptr);
        }

        /// Returns the portion of the buffer available for writing
        pub fn write_buf(&mut self) -> &mut [u8] {
            &mut self.buf[self.write_ptr..]
        }

        /// Advances the write position by `amount` bytes
        pub fn advance(&mut self, amount: usize) {
            self.write_ptr += amount;
            debug_assert!(self.write_ptr <= self.buf.len())
        }

        /// Moves any unread bytes to the start of the buffer,
        /// creating more space for writing new data
        pub fn backshift(&mut self) {
            self.buf.copy_within(self.read_ptr..self.write_ptr, 0);
            self.write_ptr -= self.read_ptr;
            self.read_ptr = 0;
        }
    }
}
