# cfit env (coarse-to-fine Paris 4 unwrap): desk forlindesk2 or forlindesk (fd). Sourced by the cfit scripts.
if [ -d /vesuvius/ufsm/paris4/fields/wind96 ]; then HOST=desk; F96=${F96:-/vesuvius/ufsm/paris4/fields/wind96}; Q=${Q:-/vesuvius/ufsm/gt/paris4-q9p6/q-9p6.zarr}; U=${U:-/vesuvius/ufsm/cfit/src/build/ufsm}
else HOST=fd; F96=${F96:-/vesuvius/ufsm/p4u/fields/wind96}; Q=${Q:-/vesuvius/ufsm/wgate/gt/paris4-q9p6/q-9p6.zarr}; U=${U:-/vesuvius/ufsm/cfit/src/build86/ufsm}; fi
P=/vesuvius/ufsm/cfit; MK=$(dirname "${BASH_SOURCE[0]}")/../wbench/mkcrop.sh
Y0=608; NY=3328; X0=352; NX=3520   # the union crop of material (field cells)
memok() { while [ $(free -g | awk '/Mem:/ {print $7}') -lt ${NEED:-55} ]; do sleep 30; done; }   # wait for NEED GB available
