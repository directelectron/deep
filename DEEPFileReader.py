# -*- coding: utf-8 -*-
"""DEEP file format reader (version 1.01).

Last modified on 27 May 2022 by Benjamin Bammes (bbammes@directelectron.com).

This class reads Detected Electron Event Puddle (DEEP) data files, which encodes
detection events on a pixelated detector, including the size, shape, and
all pixel intensities of the detection event. DEEP data can be used to implement
custom electron counting algorithms.

DEEP was developed by Direct Electron LP (San Diego, CA). Version 1.00 was
finalized on 21 January 2022. For more information email Benjamin Bammes,
Director of R&D, at bbammes@directelectron.com.

There are several other formats for encoding detection events in electron
microscopy, including the electron event representation (EER) by Thermo Fisher
Scientific (Waltham, MA) and ReCoDe formats by Datta, et al. (National
University of Singapore, https://doi.org/10.1038/s41467-020-20694-z). Unlike EER
and ReCoDe L4-compression formats, the DEEP format includes each detection
event’s shape and intensity information. Unlike the ReCoDe L1-compression
format, the DEEP format groups pixel information into individual electron
detection events instead of simply encoding all non-zero pixel intensities in a
raster format for each frame.

"""

import argparse
import bitarray
import bitarray.util
import math
import numpy
import os
import os.path
import scipy.ndimage.measurements
import sys
import tifffile

class DEEPFileReader():
	"""Reads a DEEP data file with the specified filename."""
	
	endianness = 'little'
	chunk_bytes = 1024
	
	def __init__(self, filename):
		"""Opens the file with the specified filename and reads the header.
		
		Parameters
		----------
		filename : str
			Filename (with path if not in the current working directory) of the
			DEEP file to open.
		
		"""
		
		self.file_name = filename
		self.file_object = open(self.file_name, 'rb')
		self.file_eof = False
		
		self.file_valid = False
		self.file_version = 0
		self.frame_width = 0
		self.frame_height = 0
		self.frame_bitdepth = 0
		self.frame_count = 0
		self.frame_index = -1
		self.pixel_index_bits = 0
		self.data_buffer = bitarray.bitarray()
		
		header_identifier = int.from_bytes(self.file_object.read(4), byteorder=DEEPFileReader.endianness)
		self.file_valid = header_identifier == 13240
		
		if self.file_valid:
			self.file_version = int.from_bytes(self.file_object.read(2), byteorder=DEEPFileReader.endianness)
			self.file_valid = self.file_version == 1
		
		if self.file_valid:
			self.frame_width = int.from_bytes(self.file_object.read(4), byteorder=DEEPFileReader.endianness)
			self.frame_height = int.from_bytes(self.file_object.read(4), byteorder=DEEPFileReader.endianness)
			self.frame_bitdepth = int.from_bytes(self.file_object.read(2), byteorder=DEEPFileReader.endianness)
			self.frame_count = int.from_bytes(self.file_object.read(4), byteorder=DEEPFileReader.endianness)
			header_remainder = self.file_object.read(108)
			self.pixel_index_bits = math.ceil(math.log2(self.frame_width * self.frame_height))
			self.file_eof = False
	
	def __del__(self):
		"""Closes the file and resets instance attributes.
		
		"""
		
		self.file_object.close()
		
		self.file_name = ''
		self.file_eof = True
		
		self.file_valid = False
		self.file_version = 0
		self.frame_width = 0
		self.frame_height = 0
		self.frame_bitdepth = 0
		self.frame_count = 0
		self.frame_index = -1
		self.pixel_index_bits = 0
		self.data_buffer.clear()
	
	def _remove(self, length):
		"""Removes the specified bit length from the beginning of the buffer.
		
		"""
		
		for i in range(length):
			self.data_buffer.pop(0)
	
	def _read_and_remove(self, length):
		"""Reads the specified bit length from the beginning of the buffer and removes them from the buffer.
		
		Returns
		-------
		int
			The integer value represented by the specified number of bits.
		
		"""
		
		value = bitarray.util.ba2int(self.data_buffer[:length])
		self._remove(length)
		return value
	
	def is_valid(self):
		"""Returns whether the file was loaded successfully.
		
		Returns
		-------
		bool
			True if the file loaded successfully.
		
		"""
		
		return self.file_valid
	
	def get_number_of_frames(self):
		"""Returns the number of frames in the loaded DEEP file.
		
		Returns
		-------
		int
			Total number of frames encoded in the DEEP file.
		
		"""
		
		return self.frame_count
	
	def get_next_event(self):
		"""Returns the next event puddle.
		
		Returns
		-------
		int
			The frame index (0-based) of the event., , y coordinate of
			the top-left corner of the returned 2D array within the frame, and
			the 2D array containing the real-space representation of the event
			puddle on a zero-valued background. If no more events are found,
			then this will be -1.
		int
			The x coordinate of the top-left corner of the returned 2D array
			within the frame.
		int
			The y coordinate of the top-left corner of the returned 2D array
			within the frame.
		numpy.array
			The 2D array containing the real-space representation of the event
			puddle on a zero-valued background..
		"""
		
		# Read more data to the buffer if the buffer is not long enough
		if (len(self.data_buffer) < (DEEPFileReader.chunk_bytes * 8)):
			if (not self.file_eof):
				file_string = self.file_object.read(DEEPFileReader.chunk_bytes)
				if (len(file_string) > 0):
					self.data_buffer.frombytes(file_string)
				if (len(file_string) < DEEPFileReader.chunk_bytes):
					self.file_eof = True
		
		# Check for frame padding code
		if ((self.data_buffer.count(1,0,39) == 39) and (self.data_buffer.count(1,0,40) == 39)):
			self._remove(40)
			try:
				first_one = self.data_buffer.index(1)
			except ValueError:
				return -1, 0, 0, numpy.zeros([16,16], dtype=int)
			self._remove(first_one)
		
		# Check for frame start code
		if (self.data_buffer.count(1,0,40) == 40):
			self.frame_index += 1
			self._remove(40)
		
		# Check for empty buffer
		if (len(self.data_buffer) < 1):
			return (-1, -1, -1, False)
		
		# Get the event's location on the frame
		pixel_index = self._read_and_remove(self.pixel_index_bits)
		origin_y = int(math.floor(pixel_index / self.frame_width))
		origin_x = int(pixel_index - origin_y * self.frame_width)
		
		# Get the number of rows for the event
		row_count = self._read_and_remove(4) + 1
		
		# Create the event as an empty numpy array
		event_array = numpy.zeros([row_count,16], dtype=int)
		
		# Load the event intensities
		row_width_max = 0
		for row_index in range(row_count):
			row_offset = self._read_and_remove(4)
			row_length = self._read_and_remove(4)
			for pixel_index in range(row_length):
				event_array[row_index,row_offset + pixel_index] = self._read_and_remove(self.frame_bitdepth)
			row_width = row_offset + row_length
			if (row_width > row_width_max) :
				row_width_max = row_width
			
		#if self.frame_index == 13 and origin_x == 1 and origin_y == 117:
		#	print('%i  %i  %i  %i' % (origin_x, origin_y, row_width_max, row_count))
		#	print(event_array[:,:row_width_max])
		
		# Return the event
		return self.frame_index, origin_x, origin_y, event_array[:,:row_width_max]

def main(args):
	"""Generate a TIF stack containing the events encoded in the specified DEEP file.
	
	Returns
	-------
	int
		0 if successful, non-zero on error.
	"""
	
	# Ensure all arguments are valid
	if (args.ignore < 0):
		print('ERROR: Ignore value %i is invalid (must not be negative).' % args.ignore)
		return 1
	if (args.number < 1):
		print('ERROR: Number value %i is invalid (must be greater than zero).' % args.number)
		return 1
	if (args.size < 17):
		print('ERROR: Size value %i is invalid (must be at least 17).' % args.size)
		return 1
	if (not (args.size % 2)):
		print('ERROR: Size value %i is invalid (must be odd).' % args.size)
		return 1
	
	# Ensure input file exists
	if (not os.path.exists(args.inputfile)):
		print('ERROR: Input file %s does not exist.' % args.inputfile)
		return 1
	
	# Open DEEP file
	deep_file = DEEPFileReader(args.inputfile)
	if (not deep_file.is_valid()):
		print('ERROR: Input file %s is not a valid DEEP file.' % args.inputfile)
		return 1
	
	# Check if the output file already exists
	if (os.path.exists(args.outputfile)):
		print('ERROR: Output file %s already exists.' % args.outputfile)
		return 1
	
	# Create a numpy array to hold results
	events = numpy.zeros([args.number,args.size,args.size], dtype=numpy.uint16)
	events_middle = (args.size - 1) / 2
	
	# Create a numpy array to hold summed output
	usesumframe = False
	sumframe = numpy.zeros([1,1,1], dtype=numpy.float32)
	if args.xframesize > 0 and args.yframesize > 0 :
		usesumframe = True
		sumframe = numpy.zeros([1,args.xframesize,args.yframesize], dtype=numpy.float32)
	
	# Show status
	framecount = deep_file.get_number_of_frames()
	print('Reading a maximum of %i events from %i frames...' % (args.number, framecount))
	
	# Load all events
	previous_frame_index = -1
	events_ignored = 0
	events_number = 0
	while (events_number < args.number):
		event_frame_index, event_origin_x, event_origin_y, event_pixels = deep_file.get_next_event()
		if (event_frame_index < 0):
			break
		if (events_ignored < args.ignore):
			events_ignored += 1
			continue
		
		# Skip zero-valued event
		if (numpy.sum(event_pixels) < 0.1):
			continue
		
		# Get event size and centroid
		event_height, event_width = numpy.shape(event_pixels)
		event_center_y, event_center_x = scipy.ndimage.measurements.center_of_mass(event_pixels)
		
		# Calculate the minimum x coordinate for placing the event in a box
		event_min_x = 0
		boxedevent_min_x = int(events_middle - math.floor(event_center_x + 0.5 ))
		if boxedevent_min_x < 0:
			event_min_x = 0 - boxedevent_min_x
			boxedevent_min_x = 0
			
		# Calculate the maximum x coordinate for placing the event in a box
		event_max_x = event_width
		boxedevent_max_x = boxedevent_min_x + (event_width - event_min_x)
		if boxedevent_max_x > args.size:
			boxedevent_max_x = args.size
			event_max_x = (boxedevent_max_x - boxedevent_min_x) + event_min_x
		if event_max_x > event_width:
			event_max_x = event_height
			boxedevent_max_x = (event_max_x - event_min_x) + boxedevent_min_x
		
		# Calculate the minimum y coordinate for placing the event in a box
		event_min_y = 0
		boxedevent_min_y = int(events_middle - math.floor(event_center_y + 0.5 ))
		if boxedevent_min_y < 0:
			event_min_y = 0 - boxedevent_min_y
			boxedevent_min_y = 0
		
		# Calculate the maximum y coordinate for placing the event in a box
		event_max_y = event_height
		boxedevent_max_y = boxedevent_min_y + (event_height - event_min_y)
		if boxedevent_max_y > args.size:
			boxedevent_max_y = args.size
			event_max_y = (boxedevent_max_y - boxedevent_min_y) + event_min_y
		if event_max_y > event_height:
			event_max_y = event_height
			boxedevent_max_y = (event_max_y - event_min_y) + boxedevent_min_y
		
		# Check the size
		if ((boxedevent_max_x - boxedevent_min_x) != (event_max_x - event_min_x)) or ((boxedevent_max_y - boxedevent_min_y) != (event_max_y - event_min_y)):
			print('Event Info: %i  %i  %i' % (event_frame_index, event_origin_x, event_origin_y))
			print('Center: %f %f' % (event_center_x, event_center_y))
			print('Event X: %i  %i    Box X: %i  %i' % (event_min_x, event_max_x, boxedevent_min_x, boxedevent_max_x))
			print('Event Y: %i  %i    Box Y: %i  %i' % (event_min_y, event_max_y, boxedevent_min_y, boxedevent_max_y))
			print(event_pixels)
			print('')
		
		# Place the event in a box
		events[events_number,boxedevent_min_y:boxedevent_max_y,boxedevent_min_x:boxedevent_max_x] = event_pixels[event_min_y:event_max_y,event_min_x:event_max_x]
		events_number += 1
		
		# Add the event to the accumulating sum
		if usesumframe:
			sumframe[0,event_origin_y:event_origin_y + event_height,event_origin_x:event_origin_x + event_width] += event_pixels
	
	# Save results
	save_number = events_number
	if args.outputnumber < save_number :
		save_number = args.outputnumber
	tifffile.imsave(args.outputfile, events[:save_number,:,:])
	print('Saved %i events to %s.' % (save_number, args.outputfile))
	if usesumframe:
		tifffile.imsave(args.outputfile[:-4] + '_sum.tif', sumframe)
		print('Saved %i events summed output to %s.' % (events_number, args.outputfile[:-4] + '_sum.tif'))
	
	# Show progress
	print('Done.')
	
	# Return success
	return 0

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Converts a DEEP file into a TIF stack of centered events.\nCopyright (c) 2022 Direct Electron LP (San Diego, CA USA).\nLicensed under GNU General Public License v2.0.')
	parser.add_argument('inputfile', metavar='input', type=str, help='Filename of the DEEP file to process.')
	parser.add_argument('outputfile', metavar='output', type=str, help='Filename for the output TIFF stack.')
	parser.add_argument('-i', '--ignore', type=int, default=0, help='Number of events at the beginning of the file to ignore.')
	parser.add_argument('-n', '--number', type=int, default=10000000, help='Number of events to process (default=10000000).')
	parser.add_argument('-o', '--outputnumber', type=int, default=10000, help='Number of events to output (default=10000).')
	parser.add_argument('-s', '--size', type=int, default=17, help='Output width and height (default=17).')
	parser.add_argument('-x', '--xframesize', type=int, default=0, help='Width of the summed output image.')
	parser.add_argument('-y', '--yframesize', type=int, default=0, help='Height of the summed output image.')
	args = parser.parse_args()
	sys.exit(main(args))
