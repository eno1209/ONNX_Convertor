import onnx
import onnx.utils
from onnx import optimizer
import sys
import argparse
import logging

from tools import eliminating
from tools import fusing
from tools import replacing
from tools import other
from tools import special
from tools import combo
# from tools import temp

# Main process
# Argument parser
parser = argparse.ArgumentParser(description="Optimize an ONNX model for Kneron compiler")
parser.add_argument('in_file', help='input ONNX FILE')
parser.add_argument('-o', '--output', dest='out_file', type=str, help="ouput ONNX FILE")
parser.add_argument('-D', '--debug', action='store_true', default=False, help='set logging level to DEBUG')
parser.add_argument('--bgr', type=bool, nargs='?', const=True, default=False, help="set if the model is trained in BGR mode")
parser.add_argument('--norm', type=bool, nargs='?', const=True, default=False, help="set if you have the input -0.5~0.5")
parser.add_argument('--add-bn-on-skip', type=bool, dest='bn_on_skip', nargs='?', const=True, default=False, help="set if you want to add BN on skip branches")

args = parser.parse_args()

# If in debug mode, output debug message
if args.debug:
  logging.basicConfig(level=logging.DEBUG)

if args.out_file is None:
    outfile = args.in_file[:-5] + "_polished.onnx"
    logging.debug('outfile will be created')
else:
    outfile = args.out_file
    logging.debug('outfile will be overwritted')

# Polish model in v1.4.1 includes:
#    -- nop
#    -- eliminate_identity
#    -- eliminate_nop_transpose
#    -- eliminate_nop_pad
#    -- eliminate_unused_initializer
#    -- fuse_consecutive_squeezes
#    -- fuse_consecutive_transposes
#    -- fuse_add_bias_into_conv
#    -- fuse_transpose_into_gemm

# Basic model organize
if args.in_file is None:
    logging.error('did not find input file')
else:
    m = onnx.load(args.in_file)
    logging.debug('input file successfully loaded')
# temp.weight_broadcast(m.graph)
m = combo.preprocess(m)
logging.debug('model was preprocessed - see combo.preprocess for process details')
# temp.fuse_bias_in_consecutive_1x1_conv(m.graph)

# Add BN on skip branch
if args.bn_on_skip:
    other.add_bn_on_skip_branch(m.graph)
    logging.debug('BN added on skip branch')

# My optimization
m = combo.common_optimization(m)
logging.debug('model was optimized - see combo.common_optimization for optimization details')
# Special options
if args.bgr:
    special.change_input_from_bgr_to_rgb(m)
    logging.debug('model was changed from BGR to RGB')
if args.norm:
    special.add_0_5_to_normalized_input(m)
    logging.debug('model was normalized')

# Postprocessing
m = combo.postprocess(m)
logging.debug('model was postprocessed')
onnx.save(m, outfile)
logging.info('model is saved to output file', outfile)
