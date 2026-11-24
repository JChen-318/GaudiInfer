#!/bin/sh
# Lists TPC kernels that use the special-function tables (LOOKUP*, CALC_FP_SPECIAL): only Synapse programs them, so
# under the Direct engine those instructions page-fault the device. Usage: check_direct_kernels.sh <objdir> <kernel>...
dir=$1; shift
bad=0
for k in "$@"; do
    [ -f "$dir/$k.o" ] || { echo "$k: missing"; bad=1; continue; }
    n=$(tpc-llvm-objdump -d --mcpu=gaudi2 "$dir/$k.o" | grep -ciE "lookup|calc_fp_special")
    if [ "$n" != 0 ]; then echo "$k: $n table instructions"; bad=1; fi
done
[ $bad = 0 ] && echo "all $# kernels table-free"
exit $bad
